import os

from .post import IpynbPost


def build(config):
    for filename in config.filenames:
        filename = os.path.join(config.repos_dir, filename)
        post = IpynbPost.from_file(filename)
        post_destination = get_post_destination(post, config.posts_dir)
        static_files = remap_static_files(post, config.assets_dir)
        with open(post_destination, 'w') as f:
            f.write(post.content)
        for static_filename, data in static_files.items():
            try:
                os.makedirs(os.path.dirname(static_filename))
            except FileExistsError:
                pass
            with open(static_filename, 'wb') as f:
                f.write(data)


def get_post_destination(post, posts_dir):
    date = post.date_created
    title = hyphenate(post.title)
    filename = f'{date.year}-{date.month}-{date.day}-{title}.md'
    return os.path.join(posts_dir, filename)


def remap_static_files(post, assets_dir):
    static_files = {}
    for filename, data in post.static_files.items():
        assets_filename = os.path.join(assets_dir, hyphenate(post.title), filename)
        post.content = post.content.replace(filename,
            f'{{{{ "/{assets_filename}" | asbolute_url }}}}')
        static_files[assets_filename] = data
    return static_files


def hyphenate(s):
    return s.replace(' ', '-')
