pages/pkg: src
	wasm-pack build --target no-modules
	- rm pages/pkg -r
	mv pkg pages

test: pages/pkg
	cd pages && python3 -m http.server

.PHONY: test
