pages/pkg: src
	wasm-pack build --target no-modules
	- rm pages/pkg -r
	mv pkg pages

pages/model.bin: model.bin
	cp model.bin pages/

test: pages/pkg pages/model.bin
	cd pages && python3 -m http.server

.PHONY: test
