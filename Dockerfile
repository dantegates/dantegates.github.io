FROM jekyll/builder

ADD . /srv/jekyll

RUN jekyll build --future

ENTRYPOINT ["jekyll", "server"]