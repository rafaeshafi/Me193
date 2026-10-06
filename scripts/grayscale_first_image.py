"""Save a grayscale copy of an image, from a web page, a URL, or a local file.

Usage:
    python3 scripts/grayscale_first_image.py <source> [output.png]

<source> may be a page URL (the first <img> is used), a direct image URL,
or a path to a local image file.  Only the standard library plus Pillow
is required.
"""

import io
import os
import sys
from html.parser import HTMLParser
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from PIL import Image

UA = "Mozilla/5.0 (compatible; grayscale-script/1.0)"


class ImageSources(HTMLParser):
    """Collect <img> srcs in page order, with og:image as a fallback."""

    def __init__(self):
        super().__init__()
        self.imgs = []
        self.og = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "img":
            src = a.get("src") or a.get("data-src")
            if src:
                self.imgs.append(src)
        elif tag == "meta" and self.og is None and a.get("property") == "og:image":
            self.og = a.get("content")

    @property
    def srcs(self):
        return self.imgs + ([self.og] if self.og else [])


def fetch(url):
    with urlopen(Request(url, headers={"User-Agent": UA}), timeout=30) as r:
        return r.read(), r.headers.get_content_type(), r.geturl()


def first_image(source):
    """Return (src, PIL image) for a local file, a direct image, or a page."""
    if os.path.exists(source):
        return source, Image.open(source)

    body, ctype, final_url = fetch(source)
    if ctype.startswith("image/"):
        return final_url, Image.open(io.BytesIO(body))

    parser = ImageSources()
    parser.feed(body.decode("utf-8", "replace"))
    if not parser.srcs:
        raise SystemExit(f"No <img> found on {final_url}")

    for src in parser.srcs:
        src = urljoin(final_url, src)
        try:
            # Skip what Pillow cannot decode (SVG logos, tracking pixels, ...).
            return src, Image.open(io.BytesIO(fetch(src)[0]))
        except Exception:
            continue
    raise SystemExit(f"No decodable image among {len(parser.srcs)} found on {final_url}")


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    source = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "first_image_gray.png"

    src, img = first_image(source)
    print(f"image: {src}")

    gray = img.convert("L")
    gray.save(out)
    print(f"saved:  {out}  ({gray.width}x{gray.height})")


if __name__ == "__main__":
    main()