import os

from .config import POSTS


def build(posts=POSTS):
    for post in posts:
        if os.path.exists(post.target) and not post.rebuild:
            continue
        print(f'building {post.title}')
        with open(post.target, 'w') as f:
            f.write(post.post.content)
        for static_filename, data in post.static_files.items():
            try:
                os.makedirs(os.path.dirname(static_filename))
            except FileExistsError:
                pass
            with open(static_filename, 'wb') as f:
                f.write(data)


if __name__ == '__main__':
    build()
