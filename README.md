# UA-Bench — Project Page

Project page for the paper **UA-Bench**, built with the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template) (adopted from the [Nerfies](https://nerfies.github.io/) page).

- **Live page:** https://sunzhe12.github.io/ua-bench/
- **Source repo:** https://github.com/sunzhe12/ua-bench

## Structure

GitHub Pages is configured to publish **only `docs/`**, so code and data stay
off the website.

```
.
├── README.md                   # 本文件
├── docs/                       # ← Pages 只发布这个目录
│   ├── index.html              # 整个页面，所有内容都在这里改
│   ├── .nojekyll               # 告诉 Pages 不要跑 Jekyll
│   └── static/
│       ├── css/                # Bulma + FontAwesome + 主题样式
│       ├── js/                 # 轮播、复制按钮等交互
│       ├── images/
│       │   ├── teaser.jpg      # 首页主图（论文 Figure 1）
│       │   ├── figure1~4.png   # 结果轮播图（占位，轮播当前已注释）
│       │   ├── social_preview.jpg
│       │   └── favicon.ico     # 站点图标（占位）
│       ├── pdfs/paper.pdf      # 论文 PDF（占位）
│       └── videos/             # 演示视频，>10MB 建议传 B 站 / YouTube
├── src/                        # 代码放这里，不会被 Pages 发布
└── data/                       # 数据集不进 git，见 data/README.md
```

## Fill in your content

### Done

- [x] Paper title (title, `<h1>`, Open Graph, BibTeX, meta)
- [x] Author list (12 authors; first author linked, corresponding author marked <sup>&sect;</sup>)
- [x] Institution (Southeast University) and venue (Under Review)
- [x] Abstract, verbatim from the manuscript
- [x] BibTeX entry (`@misc`, key `sun2026uabench`)
- [x] `docs/static/images/teaser.jpg` — paper Figure 1, downscaled 11520x6480 → 2000x1125 (527 KB)
- [x] `docs/static/images/social_preview.jpg` — 1200x630, cropped from the teaser

### Remaining

- [ ] `docs/static/pdfs/paper.pdf` → replace the placeholder with the real paper
- [ ] arXiv ID → uncomment the arXiv button in `docs/index.html`
- [ ] `docs/static/images/favicon.ico` → your own favicon
- [ ] Result figures → drop into `docs/static/images/figure1~4.png`, uncomment the Results carousel
- [ ] Teaser caption (removed for now — add one under the image if you want it)

Optional blocks are commented out in `docs/index.html` — uncomment to enable:
video teaser, YouTube presentation, video carousel, poster PDF, dataset /
supplementary buttons, and the "More Works" dropdown.

## GitHub Pages

Already enabled — **Deploy from a branch → `main` → `/docs`**, HTTPS enforced.

- Live page: https://sunzhe12.github.io/ua-bench/
- Every push to `main` redeploys automatically.
- Only `docs/` is published, so anything under `src/` or `data/` never reaches
  the website.

To reconfigure: Settings → Pages → *Build and deployment* → Source.

> If you ever move the page out of `docs/`, remember to update the Pages folder
> too, otherwise the site goes 404.

> The repo was renamed from `UA_bench` to `ua-bench` — lowercase is the
> convention for Pages URLs. GitHub redirects the old address automatically.

## Tips

- Compress images with [TinyPNG](https://tinypng.com); ~1920px wide is plenty.
- Host videos larger than 10MB on YouTube / Bilibili and embed them.
- During double-blind review, keep the page anonymous (no author names or
  affiliations) until the rebuttal phase.

## License

Page design licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/); parts adopted from the [Nerfies](https://nerfies.github.io/) project page. Keep the footer attribution.
