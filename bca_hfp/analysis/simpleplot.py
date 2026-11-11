# -*- coding: utf-8 -*-
"""Minimal Pillow plotting helpers (no matplotlib in this runtime)."""
from PIL import Image, ImageDraw, ImageFont

FONT_R = "C:/Windows/Fonts/arial.ttf"
FONT_B = "C:/Windows/Fonts/arialbd.ttf"


class Panel:
    def __init__(self, img, draw, rect, xlim, ylim, fs=22):
        self.img, self.d = img, draw
        self.x0, self.y0, self.w, self.h = rect
        self.xlim, self.ylim = xlim, ylim
        self.fs = fs

    def X(self, v):
        return self.x0 + (v - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * self.w

    def Y(self, v):
        return self.y0 + self.h - (v - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * self.h

    def font(self, size=None, bold=False):
        return ImageFont.truetype(FONT_B if bold else FONT_R, size or self.fs)

    def frame(self):
        d = self.d
        for xy in ([(self.x0, self.y0 + self.h), (self.x0 + self.w, self.y0 + self.h)],
                   [(self.x0, self.y0), (self.x0, self.y0 + self.h)]):
            d.line(xy, fill=(0, 0, 0), width=3)

    def xticks(self, vals, fmt="{:.1f}", ticklen=12, label_fs=None, bold=False):
        d = self.d
        for v in vals:
            x = self.X(v)
            d.line([(x, self.y0 + self.h), (x, self.y0 + self.h + ticklen)], fill=(0, 0, 0), width=3)
            t = fmt.format(v)
            f = self.font(label_fs or self.fs, bold)
            w = d.textlength(t, font=f)
            d.text((x - w / 2, self.y0 + self.h + ticklen + 8), t, fill=(0, 0, 0), font=f)

    def yticks(self, vals, fmt="{:.0f}", ticklen=12, label_fs=None, bold=False):
        d = self.d
        for v in vals:
            y = self.Y(v)
            d.line([(self.x0 - ticklen, y), (self.x0, y)], fill=(0, 0, 0), width=3)
            t = fmt.format(v)
            f = self.font(label_fs or self.fs, bold)
            w = d.textlength(t, font=f)
            d.text((self.x0 - ticklen - 10 - w, y - (label_fs or self.fs) / 2 - 2), t, fill=(0, 0, 0), font=f)

    def xlabel(self, s, pad=78, fs=None, bold=True):
        f = self.font(fs or int(self.fs * 1.25), bold)
        lines = s.split("\n")
        widths = [self.d.textlength(t, font=f) for t in lines]
        lh = (fs or int(self.fs * 1.25)) * 1.25
        for i, t in enumerate(lines):
            self.d.text((self.x0 + self.w / 2 - widths[i] / 2, self.y0 + self.h + pad + i * lh),
                        t, fill=(0, 0, 0), font=f)

    def ylabel(self, s, pad=120, fs=None, bold=True):
        f = self.font(fs or int(self.fs * 1.25), bold)
        lines = s.split("\n")
        widths = [self.d.textlength(t, font=f) for t in lines]
        lh = (fs or int(self.fs * 1.25)) * 1.25
        tmp = Image.new("RGBA", (2400, int(lh * len(lines)) + 20), (255, 255, 255, 0))
        td = ImageDraw.Draw(tmp)
        for i, t in enumerate(lines):
            td.text(((max(widths) - widths[i]) / 2, i * lh), t, fill=(0, 0, 0), font=f)
        bbox = tmp.getbbox()
        crop = tmp.crop(bbox).rotate(90, expand=True)
        self.img.paste(crop, (int(self.x0 - pad), int(self.y0 + self.h / 2 - crop.size[1] / 2)), crop)

    def circle(self, x, y, r=9, fill=(31, 92, 155), outline=None):
        X, Y = self.X(x), self.Y(y)
        self.d.ellipse([X - r, Y - r, X + r, Y + r], fill=fill, outline=outline)

    def line(self, pts, fill=(0, 0, 0), width=3, dash=None):
        P = [(self.X(a), self.Y(b)) for a, b in pts]
        if dash is None:
            self.d.line(P, fill=fill, width=width)
        else:
            # crude dashed line
            import math
            for i in range(len(P) - 1):
                x1, y1 = P[i]; x2, y2 = P[i + 1]
                L = math.hypot(x2 - x1, y2 - y1); n = max(1, int(L / dash[0]))
                for k in range(n):
                    t0 = k / n; t1 = (k + 0.5) / n
                    self.d.line([(x1 + (x2 - x1) * t0, y1 + (y2 - y1) * t0),
                                 (x1 + (x2 - x1) * t1, y1 + (y2 - y1) * t1)], fill=fill, width=width)

    def bar(self, xa, xb, ytop, fill=(150, 190, 220), outline=(60, 90, 120)):
        self.d.rectangle([self.X(xa), self.Y(ytop), self.X(xb), self.Y(0)], fill=fill, outline=outline, width=2)

    def text(self, x, y, s, fs=None, bold=False, anchor="la", fill=(0, 0, 0)):
        f = self.font(fs or self.fs, bold)
        lh = (fs or self.fs) * 1.25
        for i, t in enumerate(s.split("\n")):
            self.d.text((x, y + i * lh), t, fill=fill, font=f, anchor=anchor)


def save(img, path, scale=1):
    img.save(path, dpi=(300, 300))
    print("saved", path, img.size)
