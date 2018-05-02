FROM jekyll/builder

ADD . /srv/jekyll

RUN jekyll build

CMD ["jekyll", "server"]