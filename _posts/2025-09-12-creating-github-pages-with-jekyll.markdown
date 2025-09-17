---
layout: post
title:  "Creating Github Pages with Jekyll"
date:   2025-09-12 22:47:00 -0400
categories: Web
---
I created a github page using Jekyll today. Here are some concepts that are a little confusing.

# Jekyll
Jekyll is a static site generator. It is written in Ruby.

# Ruby
Ruby is a **interpreted programming language** (like Python). That's why you need to download Ruby to run Jekyll.

# Gem
A "gem" is Ruby’s word for a package or library.

```shell
gem install jekyll
```

# Bundler (bundle)
Bundler is a tool that manages a project’s gems (dependencies). 

Different projects may need different versions of gems. Bundler makes sure you use the correct ones.

Your project has a file called **Gemfile** listing the required gems (and versions). 

Run
```shell
bundle install
```
to install everything from Gemfile.

Instead of running `jekyll serve`, run
```shell
bundle exec jekyll serve
```
to use the gems specified in the Gemfile.

# Reference
[Ruby 101](https://jekyllrb.com/docs/ruby-101/)