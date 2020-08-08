FROM jekyll/builder

ADD . /srv/jekyll

RUN jekyll build

ENTRYPOINT ["jekyll", "server"]