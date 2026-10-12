# UA-Bench — Project Page

Project page for the paper **UA-Bench**, built with the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template) (adopted from the [Nerfies](https://nerfies.github.io/) page).

- **Live page:** https://sunzhe12.github.io/UA_bench/
- **Source repo:** https://github.com/sunzhe12/UA_bench

## Structure

```
.
├── index.html                  # 整个页面，所有内容都在这里改
├── .nojekyll                   # 告诉 GitHub Pages 不要跑 Jekyll
└── static
    ├── css/                    # Bulma + FontAwesome + 主题样式
    ├── js/                     # 轮播、复制按钮等交互
    ├── images/
    │   ├── teaser.jpg          # 首页主图（已放入论文 Figure 1）
    │   ├── figure1~4.png       # 结果轮播图（占位，轮播当前已注释）
    │   ├── social_preview.jpg  # 1200x630 社交分享预览图（由主图裁切）
    │   └── favicon.ico         # 站点图标（占位）
    ├── pdfs/paper.pdf          # 论文 PDF（占位）
    └── videos/                 # 放演示视频，>10MB 建议传 B 站 / YouTube 后嵌入
```

## Fill in your content

### Done

- [x] Paper title (title, `<h1>`, Open Graph, BibTeX, meta)
- [x] Author list (12 authors; first author linked, corresponding author marked <sup>&sect;</sup>)
- [x] Institution (Southeast University) and venue (Under Review)
- [x] Abstract, verbatim from the manuscript
- [x] BibTeX entry (`@misc`, key `sun2026uabench`)
- [x] `static/images/teaser.jpg` — paper Figure 1, downscaled 11520x6480 → 2000x1125 (527 KB)
- [x] `static/images/social_preview.jpg` — 1200x630, cropped from the teaser

### Remaining

- [ ] `static/pdfs/paper.pdf` → replace the placeholder with the real paper
- [ ] arXiv ID → uncomment the arXiv button in `index.html`
- [ ] `static/images/favicon.ico` → your own favicon
- [ ] Result figures → drop into `static/images/figure1~4.png`, uncomment the Results carousel
- [ ] Confirm the teaser caption matches the figure

Optional blocks are commented out in `index.html` — uncomment to enable:
video teaser, YouTube presentation, video carousel, poster PDF, dataset /
supplementary buttons, and the "More Works" dropdown.

## Enable GitHub Pages

1. Push this repo to `main`.
2. Repo → **Settings** → **Pages** → under *Build and deployment*, set
   **Source = Deploy from a branch**.
3. Branch: `main`, folder: `/ (root)` → **Save**.
4. After a minute or two the page goes live at
   https://sunzhe12.github.io/UA_bench/

Every push to `main` redeploys automatically.

> Note: the repo name contains uppercase letters (`UA_bench`). GitHub Pages
> URLs are case-insensitive and will redirect, but the convention is to use a
> lowercase repo name. If you rename the repo to `ua-bench`, update the URLs in
> `index.html` (Open Graph, `citation_pdf_url`, structured data, BibTeX).

## Tips

- Compress images with [TinyPNG](https://tinypng.com); ~1920px wide is plenty.
- Host videos larger than 10MB on YouTube / Bilibili and embed them.
- During double-blind review, keep the page anonymous (no author names or
  affiliations) until the rebuttal phase.

## License

Page design licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/); parts adopted from the [Nerfies](https://nerfies.github.io/) project page. Keep the footer attribution.
