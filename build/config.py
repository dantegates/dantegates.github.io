import os
import subprocess as sp
import urllib.request
import warnings
from datetime import datetime

from .posts import IpynbPost

POSTS = tuple()


def get_github_repo(directory):
    try:
        current = os.getcwd()
        os.chdir(directory)
        url = sp.check_output(['git', 'remote', 'get-url', 'origin'])
    except Exception:
        url = ''
    finally:
        os.chdir(current)
    return url.decode('utf-8').strip()


class PostConfigMeta(type):
    def __new__(metacls, name, bases, namespace):
        namespace['_github_repo'] = namespace.pop('_github_repo', None)
        return super().__new__(metacls, name, bases, namespace)

    @property
    def target(cls):
        name = f'{cls.date_created}-{cls.path_title}.{cls.extension}'
        return os.path.join(cls.posts_dir, name)

    @property
    def path(cls):
        return os.path.join(cls.parent_directory, cls.filename)

    @property
    def post(cls):
        stat = os.stat(cls.path)
        date_modified = datetime.fromtimestamp(stat.st_mtime)
        return cls.post_type.from_file(cls.path,
            date_created=cls.date_created, date_modified=date_modified,
            title=cls.title, github_repo=cls.github_repo, tags=cls.tags)

    @property
    def github_repo(cls):
        repo = getattr(cls, '_github_repo', None)
        if repo is None or repo is _UNSET:
            url = get_github_repo(os.path.dirname(cls.path))
        elif not cls._github_repo.startswith('https://'):
            url = f'https://github.com/dantegates/{cls._github_repo}'
        else:
            url = cls._github_repo
        with urllib.request.urlopen(url) as response:
            if response.status != 200:
                warnings.warn(f'GET {url} returned a status code of {response.status}')
        return cls._github_repo

    @property
    def static_files(cls):
        static_files = {}
        for filename, data in cls.post.static_files.items():
            assets_filename = os.path.join(cls.assets_dir, cls.path_title, filename)
            cls.post.content = cls.post.content.replace(filename,
                f'{{{{ "/{assets_filename}" | asbolute_url }}}}')
            static_files[assets_filename] = data
        return static_files

    @property
    def path_title(cls):
        return cls.title.replace(' ', '-').lower()


_UNSET = object()
class PostConfig(metaclass=PostConfigMeta):
    # set these
    filename = _UNSET
    date_created = _UNSET
    github_repo = _UNSET

    # safe defaults
    assets_dir = 'assets'
    posts_dir = '_posts'
    parent_directory = '../'
    post_type = IpynbPost
    extension = 'md'
    rebuild = False

    def __init_subclass__(cls):
        global POSTS
        POSTS += (cls,)
