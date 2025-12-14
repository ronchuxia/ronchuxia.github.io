# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository now serves as a redirect to the new personal homepage hosted at: http://xiachu-homepage.s3-website-us-east-1.amazonaws.com

Historical blog posts are preserved in the `_posts/` directory but are no longer published. Jekyll processing is disabled via the `.nojekyll` file, and visitors to ronchuxia.github.io are automatically redirected to the new S3-hosted site.

## Build and Development Commands

### Local Development
```bash
# Install dependencies (run after cloning or updating Gemfile)
bundle install

# Serve the site locally with live reload
bundle exec jekyll serve
```

The site will be available at `http://localhost:4000`. Jekyll watches for changes and automatically rebuilds.

### Generated Output
- Built site is generated in `_site/` directory
- This directory should not be edited manually or committed to git

## Content Structure

### Blog Posts
- Location: `_posts/`
- Naming convention: `YYYY-MM-DD-title-with-hyphens.markdown`
- Front matter required:
  ```yaml
  ---
  layout: post
  title: "Post Title"
  date: YYYY-MM-DD HH:MM:SS -0400
  categories: Category
  ---
  ```
- Common categories: Web, CUDA, 3D, Tools
- Posts with `TODO-` prefix are drafts and not published

### Assets
- Location: `assets/YYYY-MM-DD/` (organized by post date)
- Reference in posts: `{{ '/assets/YYYY-MM-DD/filename.ext' | relative_url }}`

### Static Pages
- `index.markdown`: Home page (uses `layout: home`)
- `about.markdown`: About page (uses `layout: page`)
- `404.html`: Custom 404 error page

## Configuration

### Site Settings
- Main config: `_config.yml`
- Site domain: `ronchuxia.github.io`
- Theme: Minima (GitHub Pages compatible)
- GitHub username: `ronchuxia`

### Dependencies
- Uses `github-pages` gem (version ~232) for GitHub Pages compatibility
- Jekyll Feed plugin for RSS/Atom feeds
- Ruby version managed via Bundler

## Architecture Notes

### Jekyll Build Process
1. Jekyll reads `_config.yml` for site-wide settings
2. Processes Markdown files with YAML front matter
3. Applies layouts from theme (Minima)
4. Generates static HTML in `_site/`
5. Categories in front matter create URL structure: `/category/YYYY/MM/DD/post-title.html`

### Theme System
- Uses Minima theme (installed via gem)
- Theme files are not in the repository (managed by gem)
- Can override theme files by creating matching file paths locally

## GitHub Pages Deployment

This repository is configured for GitHub Pages deployment:
- Deploys from the repository root
- Uses `github-pages` gem to match GitHub's build environment
- No manual build step needed; GitHub Pages builds automatically on push
- Site URL: `https://ronchuxia.github.io`

## Common Workflows

### Creating a New Post
1. Create file in `_posts/` with naming convention `YYYY-MM-DD-title.markdown`
2. Add required front matter (layout, title, date, categories)
3. Write content in Markdown
4. Add assets to `assets/YYYY-MM-DD/` if needed
5. Test locally with `bundle exec jekyll serve`

### Working with Drafts
- Prefix filename with `TODO-` to mark as draft (excluded from git)
- Remove `TODO-` prefix when ready to publish
