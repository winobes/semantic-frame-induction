gibbs_c.so: gibbs_c.c
	gcc -std=c99 -shared -Wl,-soname,examplelib -o gibbs_c.so -fPIC gibbs_c.c
