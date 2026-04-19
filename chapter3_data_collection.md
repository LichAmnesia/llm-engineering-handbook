# 第 3 章 数据采集 — 从社交媒体到结构化语料

## 3.1 数据决定上限

微调和 RAG 都是在"加工数据"。模型架构、超参数、tricks，都是在已有数据上刮油水；真正决定 Twin 质量天花板的，是你喂进去的原始语料本身。

这章讲清楚三件事：
1. 采什么：判断一份个人数据对 Twin 有没有用的三条标准。
2. 怎么采：LinkedIn / Medium / GitHub / X 四个主要来源各自的技术方案。
3. 存哪：MongoDB 作为半结构化数据湖的 schema 设计。

## 3.2 数据价值评分：三条标准

不是所有你写过的字都值得进 Twin。用这三条筛：

1. **作者一致性**：确认是你亲自写的。转发、quote tweet、团队联署都去掉——Twin 学到的风格必须唯一归属于你。
2. **语料密度**：每条至少 80 字符。一句"+1"、"lol" 这种短 reaction 贡献纯噪声。
3. **时效窗口**：3 年内。5 年前的你和现在的你文风差异可能比和别人的差异更大，远古语料会污染当前风格。

筛完之后的语料规模参考：

| 来源 | 原始条数 | 过滤后 | 有效字符 |
|------|----------|--------|----------|
| LinkedIn 帖子 | 300 | 180 | 120K |
| Medium 文章 | 40 | 40 | 350K |
| X 推文 | 5000 | 900 | 180K |
| GitHub README | 25 | 22 | 90K |
| 合计 | — | — | **~740K** |

740K 字符相当于 150K-200K tokens（中文稠密），做 LoRA 微调足够，不够做全参预训练——但我们本来就不会做全参预训练。

## 3.3 LinkedIn 爬取

LinkedIn 没有公开 API 可以拉自己的 feed，但可以用浏览器自动化工具爬登录后的页面。本书用 Playwright（headed mode，绕过常见反爬）：

```python
# feature_pipeline/crawlers/linkedin.py
from playwright.sync_api import sync_playwright
from datetime import datetime
import json

def crawl_linkedin_posts(profile_url: str, cookie_file: str) -> list[dict]:
    posts = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        ctx = browser.new_context(storage_state=cookie_file)
        page = ctx.new_page()
        page.goto(f"{profile_url}/recent-activity/all/")

        for _ in range(30):  # 往下滚 30 次
            page.mouse.wheel(0, 3000)
            page.wait_for_timeout(1500)

        cards = page.query_selector_all("div.feed-shared-update-v2")
        for c in cards:
            text_el = c.query_selector("span.break-words")
            if not text_el:
                continue
            posts.append({
                "source": "linkedin",
                "text": text_el.inner_text(),
                "url": c.get_attribute("data-urn") or "",
                "crawled_at": datetime.utcnow().isoformat(),
            })
        browser.close()
    return posts
```

实践提示：
- **cookie 必须手动 login 一次，然后 `storage_state` 导出**。不要把密码放代码里。
- **don't run headless**：LinkedIn 对 headless 浏览器返回阉割内容。headed 即可。
- **限速**：每 1.5 秒翻一次页，太快会触发 rate limit。我们在 10 万级语料上从未被封，但如果你需要爬别人的 profile，这样做违反 LinkedIn ToS——本书只爬自己的。

## 3.4 Medium 文章

Medium 的好在于有 RSS。直接读 feed：

```python
import feedparser

def crawl_medium(username: str) -> list[dict]:
    feed = feedparser.parse(f"https://medium.com/feed/@{username}")
    out = []
    for entry in feed.entries:
        out.append({
            "source": "medium",
            "text": entry.content[0].value,  # HTML
            "url": entry.link,
            "title": entry.title,
            "crawled_at": entry.published,
        })
    return out
```

RSS 只给最近 10 篇。历史文章要额外爬：登录 medium.com/me/stats，导出 archive，解析 HTML。代码在 `feature_pipeline/crawlers/medium_archive.py`，太长不贴了。

## 3.5 GitHub README 与博客

GitHub 的数据最干净，用官方 API：

```python
from github import Github

def crawl_github(username: str, token: str) -> list[dict]:
    gh = Github(token)
    user = gh.get_user(username)
    out = []
    for repo in user.get_repos():
        if repo.fork:  # 不要 fork 的 repo，那不是你写的
            continue
        try:
            readme = repo.get_readme().decoded_content.decode("utf-8")
        except Exception:
            continue
        if len(readme) < 300:
            continue
        out.append({
            "source": "github",
            "text": readme,
            "url": repo.html_url,
            "title": repo.name,
            "crawled_at": repo.updated_at.isoformat(),
        })
    return out
```

注意过滤 fork 的仓库——那是别人的 README。

## 3.6 X（Twitter）

X 的公开 API 已经不免费了。两条路：
1. **付费 API**（Basic tier 月费 200 美元，可以拿到自己的 all tweets）——推荐。
2. **用 nitter 镜像** + scraping——2026 年大部分 nitter 实例已被 ban，不稳定。

假设走 API：

```python
import tweepy

def crawl_x(user_id: str, bearer: str) -> list[dict]:
    client = tweepy.Client(bearer_token=bearer)
    tweets = tweepy.Paginator(
        client.get_users_tweets,
        id=user_id,
        max_results=100,
        exclude=["retweets", "replies"],
        tweet_fields=["created_at", "public_metrics"],
    ).flatten(limit=5000)

    return [{
        "source": "x",
        "text": t.text,
        "url": f"https://x.com/i/web/status/{t.id}",
        "crawled_at": t.created_at.isoformat(),
    } for t in tweets]
```

`exclude=["retweets", "replies"]` 保证作者一致性。

## 3.7 存进 MongoDB

MongoDB 作为数据湖：半结构化、schema 灵活、查询顺手。

schema 建议：

```python
# feature_pipeline/etl/schema.py
from pydantic import BaseModel, Field
from datetime import datetime

class RawDoc(BaseModel):
    _id: str                          # 来源 URL 的 hash，保证幂等
    source: str                       # linkedin | medium | github | x
    author: str                       # 规范化后的作者 ID
    text: str                         # 原始内容（可能含 HTML）
    clean_text: str | None = None     # ETL 之后填
    url: str
    title: str | None = None
    crawled_at: datetime
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    lang: str | None = None           # 'zh' | 'en'，后面做分语言微调时要用
```

写入（幂等）：

```python
from pymongo import MongoClient, UpdateOne

def upsert_docs(docs: list[RawDoc], mongo_uri: str, db: str = "twin"):
    cli = MongoClient(mongo_uri)
    col = cli[db]["raw_docs"]
    ops = [
        UpdateOne({"_id": d._id}, {"$set": d.model_dump()}, upsert=True)
        for d in docs
    ]
    if ops:
        col.bulk_write(ops)
```

用 URL hash 做 `_id` 的好处：重复爬取不会插重复条目，但 update 时能替换掉历史版本（比如你改了一篇 Medium 文章，下次爬就覆盖旧的）。

## 3.8 CDC：把 MongoDB 的写入实时推给 Feature Pipeline

传统做法是每天定时扫一遍 MongoDB 查新数据。更优雅的做法是用 MongoDB Change Stream（CDC）：

```python
# feature_pipeline/etl/cdc.py
from pymongo import MongoClient

def watch_raw_docs(mongo_uri: str, on_insert):
    cli = MongoClient(mongo_uri)
    col = cli["twin"]["raw_docs"]
    with col.watch([{"$match": {"operationType": "insert"}}]) as stream:
        for change in stream:
            on_insert(change["fullDocument"])
```

每条新 doc 一落地，立即触发下一步清洗+向量化。生产环境把 `on_insert` 改成往 Kafka/RabbitMQ 扔消息，让下游消费。这是 Ch4 的入口。

## 3.9 合规与伦理

- **只爬你自己的数据**。别人的 profile 即使公开，爬了用于训练也可能违反当地法规（GDPR/CCPA）和平台 ToS。
- **私人消息（WhatsApp / Signal / iMessage）导出后**：脱敏对方的姓名和联系方式再入库。
- **训练集里的 PII**（身份证号、电话、家庭住址）用 regex + presidio 库扫一遍，替换为占位符。

这些不是 nice-to-have，是你未来开源代码/发布模型时的硬性门槛。

## 3.10 本章小结

- 数据不是越多越好，筛三条标准：作者一致性、密度、时效。
- 四个主要来源（LinkedIn/Medium/GitHub/X）各自用合适的工具抓，别自己造轮子。
- 全部先落到 MongoDB 数据湖，用 change stream 触发下游。
- 合规和 PII 处理是硬门槛，不是收尾。

> **动手做**：跑 `python feature_pipeline/crawlers/main.py --sources linkedin,medium,github` 抓你自己的数据进 MongoDB。抓完用 `mongosh` 确认 `twin.raw_docs` 里有超过 500 条记录再进 Ch4。
