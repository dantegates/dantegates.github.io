from datetime import datetime
import os
import subprocess as sp

from nbconvert import MarkdownExporter
import nbformat


class Post:
    def __init__(self, *, content, static_files, date_created, date_modified,
                 title, github_repo):
        self.static_files = static_files
        self.date_created = date_created
        self.date_modified = date_modified
        self.title = title
        self.github_repo = github_repo
        self.content = f'{self._front_matter}\n\n{content}'

    @property
    def _front_matter(self):
        return '\n'.join((
            '---',
            'layout: post',
            'mathjax: true',
            f'title: {self.title}',
            f'github: {self.github_repo}',
            f'creation_date: {self.date_created}',
            f'last_modified: {self.date_modified}',
            '---',
        ))


class IpynbPost(Post):
    @classmethod
    def from_file(cls, ipynb_file):
        content, static_files = cls.load_file(ipynb_file)
        github_repo = cls.get_github_repo(ipynb_file)
        date_created, date_modified = cls.get_dates(ipynb_file)
        title = cls.get_title(ipynb_file)
        return cls(content=content, static_files=static_files,
            github_repo=github_repo, date_created=date_created,
            date_modified=date_modified, title=title)

    @staticmethod
    def load_file(ipynb_file):
        with open(ipynb_file) as f:
            nb = nbformat.reads(f.read(), as_version=4)
        exporter = MarkdownExporter()
        body, resources = exporter.from_notebook_node(nb)
        body = body.strip()
        if body.startswith('#'):
            # if md file starts with title remove it
            body = '\n'.join(body.split('\n')[1:])
        return body, resources['outputs']

    @staticmethod
    def get_dates(ipynb_file):
        stat = os.stat(ipynb_file)
        date_created = datetime.fromtimestamp(stat.st_birthtime)
        date_modified = datetime.fromtimestamp(stat.st_mtime)
        return date_created, date_modified

    @staticmethod
    def get_github_repo(ipynb_file):
        try:
            current = os.getcwd()
            os.chdir(os.path.dirname(ipynb_file))
            url = sp.check_output(['git', 'remote', 'get-url', 'origin'])
        finally:
            os.chdir(current)
        return url.decode('utf-8').strip()

    @staticmethod
    def get_title(ipynb_file):
        return os.path.basename(ipynb_file).split('.ipynb')[0].replace('-', ' ')
