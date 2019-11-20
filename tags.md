---
layout: page
title: Tags
permalink: /tags/
---

{% assign tags = site.tags | sort %}
{% for tags in tags %}
  <h3>{{ tags[0] }}</h3>
  <ul>
    {% for post in tags[1] %}
      <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %}
  </ul>
{% endfor %}
