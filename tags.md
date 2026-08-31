---
layout: page
title: Tags
permalink: /tags/
---

{% assign tags = site.tags | sort %}

{% for tag in tags %} <div class="site-tag"><a href="#{{ tag[0] | slugify }}" class="tag">{{ tag[0] }}<a/></div> {% endfor %}

{% for tag in tags %}
  <h1 id="{{ tag[0] | slugify }}">{{ tag[0] }}</h1>
  <ul>
    {% for post in tag[1] %}
      <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %}
  </ul>
{% endfor %}
