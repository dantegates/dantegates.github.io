build:
	python -m build \
	&& docker build -t dantegates.github.io . \
	&& docker run -p 4000:4000 dantegates.github.io

watch:
	docker build -t dantegates.github.io . \
	&& docker run -p 4000:4000 dantegates.github.io

clean:
	rm -r vendor
	rm build.sh

.PHONY: build
