from dataclasses import dataclass
import asyncio
import html
import ipaddress
import re
import socket
import time
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlencode, urljoin, urlparse
from xml.etree import ElementTree as ET

from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

from astrbot.api import logger
from astrbot.api.star import Context, Star, register
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str
    published: str = ""
    content: str = ""
    content_error: str = ""


def _clean_text(value: str | None) -> str:
    """Remove markup/noise from RSS titles and descriptions."""
    text = html.unescape(value or "")
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_UNSAFE_QUERY_PATTERNS = [
    (
        r"(如何|怎么|教程|方法|步骤|教我).{0,20}"
        r"(制作|自制|合成|购买|获取).{0,20}"
        r"(炸弹|炸药|毒品|冰毒|枪支|管制刀具|假证|洗钱)"
    ),
    (
        r"(盗号|撞库|勒索软件|木马|后门|免杀|ddos|钓鱼网站).{0,20}"
        r"(教程|源码|搭建|制作|实施|工具|方法)"
    ),
    (
        r"(绕过|破解).{0,20}"
        r"(付费|会员|drm|正版|版权|平台风控|实名|账号封禁|登录验证)"
    ),
    (
        r"(人肉|开盒|查身份证|查户籍|查手机号|定位他人).{0,20}"
        r"(教程|方法|工具|服务|渠道)"
    ),
    (
        r"(购买|出售|代购|求购).{0,20}"
        r"(毒品|枪支|弹药|假证|身份信息|银行卡|账号|公民个人信息)"
    ),
    (
        r"((未成年|儿童|幼女|小学生).{0,20}(色情|裸照|猥亵|性侵))|"
        r"((色情|裸照|猥亵|性侵).{0,20}(未成年|儿童|幼女|小学生))"
    ),
    (
        r"(逃避追查|销毁证据|躲避警察|逃避法律).{0,20}"
        r"(教程|方法|工具|渠道)"
    ),
]


def _query_safety_error(query: str) -> str | None:
    """Reject requests that ask for clearly illegal or harmful instructions."""
    normalized = re.sub(r"\s+", "", query.lower())
    for pattern in _UNSAFE_QUERY_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            return (
                "搜索已拒绝：请求涉及可能违法、危险、侵犯隐私或绕过平台限制的内容。"
                "插件不会执行这类搜索，也不会尝试绕开访问控制。"
            )
    return None


async def _validate_public_url(url: str) -> None:
    """Reject local, private, reserved, and non-standard web endpoints."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"不支持的链接协议：{parsed.scheme}")
    if not parsed.hostname:
        raise ValueError("链接缺少有效主机名")
    if parsed.port not in {None, 80, 443}:
        raise ValueError(f"不允许访问非标准端口：{parsed.port}")

    hostname = parsed.hostname.strip().lower().rstrip(".")
    if (
        hostname in {"localhost", "localhost.localdomain"}
        or hostname.endswith((".local", ".internal", ".lan"))
    ):
        raise ValueError("不允许访问本机或内网地址")

    try:
        addresses = [ipaddress.ip_address(hostname)]
    except ValueError:
        loop = asyncio.get_running_loop()
        try:
            infos = await loop.getaddrinfo(
                hostname,
                parsed.port or (443 if parsed.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
        except socket.gaierror as exc:
            raise ValueError(f"无法解析主机名：{hostname}") from exc
        addresses = [
            ipaddress.ip_address(info[4][0])
            for info in infos
            if info[4] and info[4][0]
        ]

    if not addresses:
        raise ValueError(f"无法解析主机名：{hostname}")
    for address in addresses:
        if not address.is_global:
            raise ValueError("不允许访问本机、内网、保留地址或云元数据地址")


def _normalize_content(text: str, max_chars: int) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:max_chars]


class _VisibleTextParser(HTMLParser):
    """Non-recursive visible-text extractor for untrusted HTML."""

    skip_tags = {
        "script",
        "style",
        "noscript",
        "svg",
        "canvas",
        "form",
        "nav",
        "footer",
        "header",
        "aside",
    }
    block_tags = {
        "article",
        "section",
        "main",
        "div",
        "p",
        "li",
        "br",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "tr",
        "td",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in self.skip_tags:
            self._skip_depth += 1
        elif self._skip_depth == 0 and tag in self.block_tags:
            self._parts.append("\n")

    def handle_startendtag(self, tag: str, attrs) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.skip_tags:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif self._skip_depth == 0 and tag in self.block_tags:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        value = " ".join(data.split())
        if value:
            self._parts.append(value)

    def text(self) -> str:
        return "\n".join(self._parts)


def _extract_html_content(raw: bytes, max_chars: int) -> str:
    charset_match = re.search(
        rb"charset\s*=\s*[\"']?([A-Za-z0-9._-]+)",
        raw[:8192],
        flags=re.IGNORECASE,
    )
    encoding = "utf-8"
    if charset_match:
        encoding = charset_match.group(1).decode("ascii", "ignore").lower()

    try:
        document = raw.decode(encoding, "replace")
    except LookupError:
        document = raw.decode("utf-8", "replace")

    parser = _VisibleTextParser()
    parser.feed(document)
    return _normalize_content(parser.text(), max_chars)


def _github_api_headers() -> dict[str, str]:
    return {
        "User-Agent": "astrbot-web-search/1.0.0",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


async def _fetch_github_repo_content(
    repo: str,
    timeout_seconds: float,
) -> SearchHit | None:
    import aiohttp

    headers = _github_api_headers()
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
        repo_url = f"https://api.github.com/repos/{repo}"
        async with session.get(repo_url, headers=headers) as response:
            if response.status != 200:
                return None
            item = await response.json()

        readme = ""
        readme_url = f"https://api.github.com/repos/{repo}/readme"
        async with session.get(
            readme_url,
            headers={
                **headers,
                "Accept": "application/vnd.github.raw+json",
            },
        ) as response:
            if response.status == 200:
                readme = _normalize_content(await response.text(), 12000)

    full_name = str(item.get("full_name") or repo).strip()
    html_url = str(item.get("html_url") or f"https://github.com/{repo}").strip()
    description = _clean_text(item.get("description"))
    language = _clean_text(item.get("language"))
    stars = int(item.get("stargazers_count") or 0)
    forks = int(item.get("forks_count") or 0)
    topics = ", ".join(map(str, item.get("topics") or []))
    default_branch = _clean_text(item.get("default_branch"))

    meta_lines = [
        f"仓库：{full_name}",
        f"链接：{html_url}",
        f"描述：{description or '无'}",
        f"语言：{language or '未知'}",
        f"Stars：{stars}，Forks：{forks}",
        f"默认分支：{default_branch or '未知'}",
    ]
    if topics:
        meta_lines.append(f"Topics：{topics}")
    if readme:
        meta_lines.extend(["", "README 正文：", readme])

    return SearchHit(
        title=f"GitHub 仓库 {full_name}",
        url=html_url,
        snippet=" | ".join(meta_lines[:6]),
        published=_clean_text(item.get("updated_at")),
        content="\n".join(meta_lines),
    )


async def _fetch_page_content(
    url: str,
    timeout_seconds: float,
    max_chars: int,
) -> str:
    import aiohttp

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"不支持的链接协议：{parsed.scheme}")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    raw: bytes | None = None
    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
        current_url = url
        for redirect_count in range(4):
            await _validate_public_url(current_url)
            async with session.get(
                current_url,
                headers=headers,
                allow_redirects=False,
            ) as response:
                if response.status in {301, 302, 303, 307, 308}:
                    location = response.headers.get("Location")
                    if not location:
                        raise RuntimeError("重定向缺少目标地址")
                    if redirect_count >= 3:
                        raise RuntimeError("重定向次数过多")
                    current_url = urljoin(current_url, location)
                    continue

                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}")
                disposition = (
                    response.headers.get("Content-Disposition") or ""
                ).lower()
                if "attachment" in disposition:
                    raise RuntimeError("拒绝下载附件，只读取公开网页正文")
                content_type = (response.headers.get("Content-Type") or "").lower()
                if (
                    "text/html" not in content_type
                    and "application/xhtml" not in content_type
                ):
                    raise RuntimeError(f"非 HTML 页面：{content_type or 'unknown'}")
                content_length = response.headers.get("Content-Length")
                if content_length:
                    try:
                        if int(content_length) > 2_000_000:
                            raise RuntimeError("页面体积过大")
                    except ValueError:
                        pass
                raw = await response.content.read(max(250_000, max_chars * 8))
                if len(raw) > 2_000_000:
                    raise RuntimeError("页面体积过大")
                break

    if raw is None:
        raise RuntimeError("未能读取网页正文")

    text = _extract_html_content(raw, max_chars)
    if len(text) < 120:
        raise RuntimeError("正文过短")
    return text


def _build_search_urls(query: str, max_results: int, region: str) -> list[str]:
    params = urlencode(
        {
            "q": query,
            "format": "rss",
            "count": max(1, min(max_results, 20)),
            "setlang": region,
            "cc": "CN" if region.lower().startswith("zh") else "US",
        }
    )
    return [
        f"https://cn.bing.com/search?{params}",
        f"https://www.bing.com/search?{params}",
    ]


def _search_headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0 Safari/537.36"
        ),
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }


async def _sogou_search(
    query: str,
    max_results: int,
    timeout_seconds: float,
) -> list[SearchHit]:
    import aiohttp
    from bs4 import BeautifulSoup

    params = urlencode({"query": query})
    url = f"https://www.sogou.com/web?{params}"
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    async with aiohttp.ClientSession(
        timeout=timeout,
        trust_env=True,
        headers=_search_headers(),
    ) as session:
        async with session.get(url, allow_redirects=True, max_redirects=3) as response:
            if response.status != 200:
                return []
            raw = await response.content.read(1_500_000)

    soup = BeautifulSoup(raw, "lxml")
    hits: list[SearchHit] = []
    seen: set[str] = set()
    for title_node in soup.select(".vr-title, .vrTitle"):
        anchor = title_node.find("a", href=True)
        if not anchor:
            continue
        title = _clean_text(anchor.get_text(" ", strip=True))
        href = urljoin("https://www.sogou.com", anchor.get("href", ""))
        if not title or not href or href in seen:
            continue
        seen.add(href)
        container = title_node.find_parent(class_=re.compile("vrwrap|result"))
        snippet = ""
        if container:
            snippet_node = container.select_one(
                ".fz-mid, .space-txt, .str-text-info, .text-layout"
            )
            snippet = _clean_text(snippet_node.get_text(" ", strip=True)) if snippet_node else ""
        hits.append(SearchHit(title=title, url=href, snippet=snippet))
        if len(hits) >= max_results:
            break
    return hits


async def _so360_search(
    query: str,
    max_results: int,
    timeout_seconds: float,
) -> list[SearchHit]:
    import aiohttp
    from bs4 import BeautifulSoup

    params = urlencode({"q": query})
    url = f"https://www.so.com/s?{params}"
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    async with aiohttp.ClientSession(
        timeout=timeout,
        trust_env=True,
        headers=_search_headers(),
    ) as session:
        async with session.get(url, allow_redirects=True, max_redirects=3) as response:
            if response.status != 200:
                return []
            raw = await response.content.read(2_000_000)

    soup = BeautifulSoup(raw, "lxml")
    hits: list[SearchHit] = []
    seen: set[str] = set()
    for title_node in soup.select(".res-list, .result, .g-title"):
        anchor = (
            title_node.find("a", href=True)
            if title_node.name != "a"
            else title_node
        )
        if not anchor:
            continue
        title = _clean_text(anchor.get_text(" ", strip=True))
        href = urljoin("https://www.so.com", anchor.get("href", ""))
        if not title or not href or href in seen:
            continue
        if href.startswith("https://www.so.com/s?"):
            continue
        seen.add(href)
        snippet_node = title_node.select_one(".res-desc, .summary, .res-rich")
        snippet = _clean_text(snippet_node.get_text(" ", strip=True)) if snippet_node else ""
        hits.append(SearchHit(title=title, url=href, snippet=snippet))
        if len(hits) >= max_results:
            break
    return hits


async def _common_search(
    query: str,
    max_results: int,
    timeout_seconds: float,
    region: str,
) -> list[SearchHit]:
    errors: list[str] = []
    sources = [
        ("Bing", _bing_search),
        ("Sogou", _sogou_search),
        ("360", _so360_search),
    ]
    for name, searcher in sources:
        try:
            if name == "Bing":
                hits = await searcher(query, max_results, timeout_seconds, region)
            else:
                hits = await searcher(query, max_results, timeout_seconds)
            if hits:
                return hits
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    if errors:
        logger.warning("通用搜索源失败：%s", "；".join(errors))
    return []


def _looks_like_github_query(query: str) -> bool:
    lowered = query.lower()
    if "github" in lowered:
        return True
    return bool(re.search(r"\b[\w.-]+/[\w.-]+\b", query))


def _looks_like_bilibili_query(query: str) -> bool:
    lowered = query.lower()
    markers = (
        "bilibili",
        "b站",
        "哔哩哔哩",
        "up主",
        "up 主",
        "up主",
        "投稿",
        "视频",
    )
    return any(marker in lowered for marker in markers)


def _looks_like_weather_query(query: str) -> bool:
    markers = (
        "天气",
        "气温",
        "温度",
        "降雨",
        "下雨",
        "下雪",
        "台风",
        "空气质量",
        "weather",
    )
    return any(marker in query.lower() for marker in markers)


def _weather_city(query: str) -> str:
    city = re.sub(
        r"(今天|今日|明天|后天|现在|实时|当前|最近|的|天气|气温|温度|"
        r"降雨|下雨|下雪|台风|空气质量|怎么样|如何|查询|查一下|帮我|"
        r"看看|多少度|几度|预报|weather|today|tomorrow|now|forecast)",
        " ",
        query,
        flags=re.IGNORECASE,
    )
    city = re.sub(r"[，。！？、,.!?：:；;（）()\[\]【】]", " ", city)
    return re.sub(r"\s+", " ", city).strip()


async def _weather_search(
    query: str,
    timeout_seconds: float,
) -> list[SearchHit]:
    import aiohttp

    city = _weather_city(query)
    if not city:
        return []

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0 Safari/537.36"
        ),
    }
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    async with aiohttp.ClientSession(
        timeout=timeout,
        trust_env=True,
        headers=headers,
    ) as session:
        async with session.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": city,
                "count": 1,
                "language": "zh",
                "format": "json",
            },
        ) as response:
            geo = await response.json() if response.status == 200 else {}
        results = geo.get("results") or []
        if not results:
            async with session.get(
                f"https://wttr.in/{urlencode({'': city})[1:]}",
                params={"format": "j1", "lang": "zh"},
            ) as response:
                if response.status != 200:
                    return []
                weather = await response.json()
            location = _clean_text(
                (((weather.get("nearest_area") or [{}])[0].get("areaName") or [{}])[0]).get("value")
            )
        else:
            place = results[0]
            location = _clean_text(place.get("name"))
            admin = _clean_text(place.get("admin1"))
            country = _clean_text(place.get("country"))
            async with session.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "current": (
                        "temperature_2m,relative_humidity_2m,apparent_temperature,"
                        "precipitation,weather_code,wind_speed_10m"
                    ),
                    "daily": (
                        "weather_code,temperature_2m_max,temperature_2m_min,"
                        "precipitation_probability_max,sunrise,sunset"
                    ),
                    "timezone": "auto",
                    "forecast_days": 3,
                },
            ) as response:
                if response.status != 200:
                    return []
                weather = await response.json()
            if admin and admin not in location:
                location = f"{admin}{location}"
            if country and country != "中国" and country not in location:
                location = f"{location}（{country}）"

    current = weather.get("current") or {}
    daily = weather.get("daily") or {}
    lines = [
        f"天气查询城市：{location or city}",
        "",
        "当前天气：",
        f"- 观测时间：{_clean_text(current.get('time'))}",
        f"- 气温：{current.get('temperature_2m')}°C",
        f"- 体感温度：{current.get('apparent_temperature')}°C",
        f"- 相对湿度：{current.get('relative_humidity_2m')}%",
        f"- 降水量：{current.get('precipitation')} mm",
        f"- 风速：{current.get('wind_speed_10m')} km/h",
        f"- 天气代码：{current.get('weather_code')}",
        "",
        "未来天气预报：",
    ]
    dates = daily.get("time") or []
    for index, date in enumerate(dates):
        lines.append(
            f"- {date}：最高 {_safe_index(daily.get('temperature_2m_max'), index)}°C，"
            f"最低 {_safe_index(daily.get('temperature_2m_min'), index)}°C，"
            f"最高降水概率 {_safe_index(daily.get('precipitation_probability_max'), index)}%，"
            f"天气代码 {_safe_index(daily.get('weather_code'), index)}"
        )

    return [
        SearchHit(
            title=f"{location or city} 实时天气与未来预报",
            url="https://open-meteo.com/",
            snippet="Open-Meteo 实时天气数据",
            content="\n".join(lines),
        )
    ]


def _safe_index(values: Any, index: int) -> Any:
    if isinstance(values, list) and 0 <= index < len(values):
        return values[index]
    return "未知"


def _bilibili_keyword(query: str) -> str:
    keyword = query
    for marker in (
        "哔哩哔哩",
        "bilibili",
        "Bilibili",
        "B站",
        "b站",
        "up主",
        "UP主",
        "up 主",
        "UP 主",
        "投稿",
        "视频",
        "搜索",
        "查一下",
        "帮我",
        "看看",
    ):
        keyword = re.sub(re.escape(marker), " ", keyword, flags=re.IGNORECASE)
    keyword = re.sub(r"[，。！？、,.!?：:；;（）()\[\]【】]", " ", keyword)
    return re.sub(r"\s+", " ", keyword).strip()


async def _bilibili_search(
    query: str,
    max_results: int,
    timeout_seconds: float,
) -> list[SearchHit]:
    import aiohttp

    keyword = _bilibili_keyword(query) or query
    params = urlencode(
        {
            "search_type": "bili_user",
            "keyword": keyword,
            "page": 1,
        }
    )
    url = f"https://api.bilibili.com/x/web-interface/wbi/search/type?{params}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0 Safari/537.36"
        ),
        "Referer": "https://search.bilibili.com/",
        "Accept": "application/json, text/plain, */*",
    }
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                raise RuntimeError(f"B站搜索接口返回 HTTP {response.status}")
            data = await response.json()

    if int(data.get("code") or 0) != 0:
        raise RuntimeError(
            f"B站搜索接口返回错误：{data.get('message') or data.get('code')}"
        )

    hits: list[SearchHit] = []
    for item in (data.get("data") or {}).get("result") or []:
        if item.get("type") != "bili_user":
            continue
        mid = item.get("mid")
        uname = _clean_text(item.get("uname"))
        if not mid or not uname:
            continue

        fans = int(item.get("fans") or 0)
        videos = int(item.get("videos") or 0)
        sign = _clean_text(item.get("usign"))
        verify = _clean_text(item.get("verify_info"))
        recent = []
        for video in item.get("res") or []:
            title = _clean_text(video.get("title"))
            arcurl = _clean_text(video.get("arcurl"))
            if title and arcurl:
                recent.append(f"{title} ({arcurl})")

        lines = [
            f"UP主：{uname}",
            f"UID：{mid}",
            f"主页：https://space.bilibili.com/{mid}",
            f"粉丝数：{fans}，投稿数：{videos}",
        ]
        if verify:
            lines.append(f"认证：{verify}")
        if sign:
            lines.append(f"简介：{sign}")
        if recent:
            lines.extend(["最近投稿：", *recent[:5]])

        hits.append(
            SearchHit(
                title=f"B站 UP主 {uname} 的主页与投稿",
                url=f"https://space.bilibili.com/{mid}",
                snippet=" | ".join(lines[:5]),
                content="\n".join(lines),
            )
        )
        if len(hits) >= max_results:
            break
    return hits


def _extract_github_repo(query: str) -> str | None:
    """Extract an owner/repo pair or a repository name from a user query."""
    direct = re.search(
        r"(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)",
        query,
        flags=re.IGNORECASE,
    )
    if direct:
        return direct.group(1).strip("./")

    pair = re.search(r"\b([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)\b", query)
    if pair:
        return pair.group(1).strip("./")

    stop_words = {
        "a",
        "about",
        "and",
        "for",
        "github",
        "in",
        "of",
        "on",
        "project",
        "repo",
        "repository",
        "search",
        "summary",
        "the",
        "to",
        "一个",
        "一下",
        "中",
        "仓库",
        "作用",
        "内容",
        "可以",
        "总结",
        "搜索",
        "这个",
        "项目",
        "帮我",
        "看看",
    }
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,}", query)
    useful = [
        token
        for token in tokens
        if token.lower() not in stop_words
    ]
    return " ".join(useful[:3]) if useful else None


async def _github_search(
    query: str,
    max_results: int,
    timeout_seconds: float,
) -> list[SearchHit]:
    import aiohttp

    repo = _extract_github_repo(query)
    if not repo:
        return []

    headers = {
        "User-Agent": "astrbot-web-search/1.0.0",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
        if "/" in repo:
            url = f"https://api.github.com/repos/{repo}"
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    return []
                item = await response.json()
                items = [item]
        else:
            params = urlencode(
                {
                    "q": repo,
                    "per_page": max(1, min(max_results, 20)),
                    "sort": "stars",
                    "order": "desc",
                }
            )
            url = f"https://api.github.com/search/repositories?{params}"
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    return []
                data = await response.json()
                items = data.get("items") or []
                repo_name = repo.lower()
                items = sorted(
                    items,
                    key=lambda item: (
                        0
                        if str(item.get("name") or "").lower() == repo_name
                        else 1,
                        -int(item.get("stargazers_count") or 0),
                    ),
                )

    hits: list[SearchHit] = []
    for item in items[:max_results]:
        full_name = str(item.get("full_name") or "").strip()
        html_url = str(item.get("html_url") or "").strip()
        if not full_name or not html_url:
            continue

        description = _clean_text(item.get("description"))
        language = _clean_text(item.get("language"))
        stars = int(item.get("stargazers_count") or 0)
        forks = int(item.get("forks_count") or 0)
        updated_at = _clean_text(item.get("updated_at"))
        topics = item.get("topics") or []
        snippet_parts = []
        if description:
            snippet_parts.append(description)
        stats = f"Stars: {stars}, Forks: {forks}"
        if language:
            stats += f", Language: {language}"
        snippet_parts.append(stats)
        if topics:
            snippet_parts.append("Topics: " + ", ".join(map(str, topics[:8])))
        hits.append(
            SearchHit(
                title=f"GitHub 仓库 {full_name}",
                url=html_url,
                snippet=" | ".join(snippet_parts),
                published=updated_at,
            )
        )

    if "/" not in repo:
        enrich_count = min(3, len(hits))
        enriched: list[SearchHit] = []
        for hit in hits[:enrich_count]:
            repo_name = hit.title.removeprefix("GitHub 仓库 ").strip()
            detailed = (
                await _fetch_github_repo_content(repo_name, timeout_seconds)
                if repo_name
                else None
            )
            enriched.append(detailed or hit)
        hits = enriched + hits[enrich_count:]
    return hits


async def _bing_search(
    query: str,
    max_results: int,
    timeout_seconds: float,
    region: str,
) -> list[SearchHit]:
    import aiohttp

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0 Safari/537.36"
        ),
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    last_error = ""

    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
        for url in _build_search_urls(query, max_results, region):
            try:
                async with session.get(url, headers=headers) as response:
                    raw = await response.read()
                    if response.status != 200:
                        last_error = f"HTTP {response.status}"
                        continue
                    try:
                        root = ET.fromstring(raw)
                    except ET.ParseError as exc:
                        last_error = f"RSS 解析失败: {exc}"
                        continue

                    hits: list[SearchHit] = []
                    seen_urls: set[str] = set()
                    for item in root.findall("./channel/item"):
                        title = _clean_text(item.findtext("title"))
                        link = _clean_text(item.findtext("link"))
                        if not title or not link or link in seen_urls:
                            continue
                        seen_urls.add(link)
                        hits.append(
                            SearchHit(
                                title=title,
                                url=link,
                                snippet=_clean_text(item.findtext("description")),
                                published=_clean_text(item.findtext("pubDate")),
                            )
                        )
                        if len(hits) >= max_results:
                            break
                    if hits:
                        return hits
                    last_error = "搜索源没有返回结果"
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"

    raise RuntimeError(last_error or "没有可用的搜索源")


@pydantic_dataclass
class WebSearchTool(FunctionTool[AstrAgentContext]):
    """Web search tool for the active AstrBot chat model."""

    max_results: int = 7
    timeout_seconds: float = 20.0
    region: str = "zh-CN"
    include_published: bool = True
    fetch_pages: bool = True
    max_pages: int = 4
    page_timeout_seconds: float = 15.0
    max_page_chars: int = 5000
    block_unsafe_queries: bool = True

    name: str = "web_search"
    description: str = (
        "联网搜索并阅读网页正文。先通过 Bing/Sogou/360/GitHub/B站/天气接口搜索候选内容，"
        "再抓取前几条页面正文，返回给模型进行总结。"
        "适用于新闻、天气、项目仓库、B站UP主、网页最新内容、实时动态等需要联网检索的问题。"
        "参数 query 为要搜索的问题或关键词。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "要搜索的问题或关键词，例如：上海今天天气",
                },
            },
            "required": ["query"],
        }
    )

    async def call(
        self,
        context: ContextWrapper[AstrAgentContext],
        **kwargs: Any,
    ) -> ToolExecResult:
        del context
        query = str(kwargs.get("query") or "").strip()
        if not query:
            return "搜索失败：没有收到要搜索的 query 参数。"
        if self.block_unsafe_queries:
            safety_error = _query_safety_error(query)
            if safety_error:
                return safety_error

        try:
            max_results = max(1, min(int(self.max_results), 20))
            timeout = max(3.0, float(self.timeout_seconds))
            hits: list[SearchHit] = []
            if _looks_like_github_query(query):
                repo = _extract_github_repo(query)
                if repo and "/" in repo:
                    hit = await _fetch_github_repo_content(repo, timeout)
                    if hit:
                        hits = [hit]
                if not hits:
                    hits = await _github_search(query, max_results, timeout)
            if not hits and _looks_like_bilibili_query(query):
                hits = await _bilibili_search(query, max_results, timeout)
            if not hits and _looks_like_weather_query(query):
                hits = await _weather_search(query, timeout)
            if not hits:
                hits = await _common_search(
                    query=query,
                    max_results=max_results,
                    timeout_seconds=timeout,
                    region=self.region or "zh-CN",
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("网页搜索调用异常")
            return f"搜索失败：{type(exc).__name__}: {exc}"

        if not hits:
            return f"搜索完成，但没有找到与 {query!r} 相关的结果。"

        if self.fetch_pages:
            import asyncio

            page_limit = max(0, min(int(self.max_pages), len(hits)))

            async def fetch(hit: SearchHit) -> None:
                if hit.content:
                    return
                try:
                    hit.content = await _fetch_page_content(
                        hit.url,
                        max(3.0, float(self.page_timeout_seconds)),
                        max(500, min(int(self.max_page_chars), 20000)),
                    )
                except Exception as exc:  # noqa: BLE001
                    hit.content_error = f"{type(exc).__name__}: {exc}"

            await asyncio.gather(*(fetch(hit) for hit in hits[:page_limit]))

        now = time.strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            f"已通过网络搜索并读取网页内容，查询：{query!r}，搜索时间：{now}",
            "请依据以下正文内容进行总结、回答和引用来源；不要只说搜索结果列表。",
            "如果正文不足，请明确说明。",
            "",
        ]
        for index, hit in enumerate(hits, 1):
            lines.append(f"{index}. {hit.title}")
            lines.append(f"   来源：{hit.url}")
            if hit.snippet:
                lines.append(f"   摘要：{hit.snippet}")
            if self.include_published and hit.published:
                lines.append(f"   发布时间：{hit.published}")
            if hit.content:
                lines.append("   正文：")
                lines.append(hit.content)
            elif hit.content_error:
                lines.append(f"   正文读取失败：{hit.content_error}")
            lines.append("")

        logger.info(
            "网页搜索完成，query=%r，来源 %d 条",
            query,
            len(hits),
        )
        return "\n".join(lines).strip()


@register(
    "astrbot_plugin_web_search",
    "ayacyann",
    "网页搜索插件：联网检索并读取网页正文，交给当前聊天模型总结",
    "1.0.0",
    "",
)
class WebSearchPlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config

        try:
            max_results = int(config.get("max_results", 7))
        except (TypeError, ValueError):
            max_results = 7
        try:
            timeout = float(config.get("timeout_seconds", 20))
        except (TypeError, ValueError):
            timeout = 20.0

        region = str(config.get("region") or "zh-CN").strip()
        include_published = bool(config.get("include_published", True))
        fetch_pages = bool(config.get("fetch_pages", True))
        try:
            max_pages = int(config.get("max_pages", 4))
        except (TypeError, ValueError):
            max_pages = 4
        try:
            page_timeout = float(config.get("page_timeout_seconds", 15))
        except (TypeError, ValueError):
            page_timeout = 15.0
        try:
            max_page_chars = int(config.get("max_page_chars", 5000))
        except (TypeError, ValueError):
            max_page_chars = 5000
        block_unsafe_queries = bool(config.get("block_unsafe_queries", True))

        tool = WebSearchTool(
            max_results=max_results,
            timeout_seconds=timeout,
            region=region,
            include_published=include_published,
            fetch_pages=fetch_pages,
            max_pages=max_pages,
            page_timeout_seconds=page_timeout,
            max_page_chars=max_page_chars,
            block_unsafe_queries=block_unsafe_queries,
        )
        self.context.add_llm_tools(tool)
        logger.info(
            "网页搜索工具已注册（使用当前 AstrBot 聊天模型，"
            "最多返回 %d 条结果，抓取 %d 个页面正文，超时 %.1fs）",
            max_results,
            max_pages,
            timeout,
        )
