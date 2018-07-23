from nbconvert import MarkdownExporter
import nbformat


POSTS = tuple()


class Post:
    def __init__(self, *, content, static_files, date_created, date_modified,
                 title, github_repo, tags=None):
        self.static_files = static_files
        self.date_created = date_created
        self.date_modified = date_modified
        self.title = title
        self.github_repo = github_repo
        self._tags = [] if tags is None else tags
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
            f'tags: {self.tags}'
            '---',
        ))

    @property
    def tags(self):
        if self._tags is not None:
            nl = '\n'
            tags = f'{nl}  - {f"{nl}  - ".join(self._tags)}{nl}'
        else:
            tags = ''
        return tags


class IpynbPost(Post):
    @classmethod
    def from_file(cls, ipynb_file, **kwargs):
        content, static_files = cls.load_file(ipynb_file)
        return cls(content=content, static_files=static_files, **kwargs)

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
