---
layout: page
title: Tags
permalink: /tags/
---

{% assign tags = site.tags | sort %}

<div class="site-tag">
{% for tag in tags %} <a href="#{{ tag[0] | slugify }}" class="tag">{{ tag[0] }}</a> {% endfor %}
</div>

{% for tag in tags %}
  <h1 id="{{ tag[0] | slugify }}">{{ tag[0] }}</h1>
  <ul>
    {% for post in tag[1] %}
      <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %}
  </ul>
{% endfor %}
