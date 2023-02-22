FROM jekyll/builder:3.8

ADD . /srv/jekyll

RUN jekyll build

ENTRYPOINT ["jekyll", "serve"]