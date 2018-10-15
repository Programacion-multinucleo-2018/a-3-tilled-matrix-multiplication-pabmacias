CC = nvcc
CFLAGS = -Wno-deprecated-gpu-targets -std=c++11
INCLUDES =
LDFLAGS =
SOURCES = matrix_mult_tile.cu
OUTF = matrix_mult_tile.exe
OBJS = matrix_mult_tile.o

$(OUTF): $(OBJS)
	$(CC) $(CFLAGS) -o $(OUTF) $< $(LDFLAGS)

$(OBJS): $(SOURCES)
	$(CC) $(CFLAGS) -c $<

rebuild: clean $(OUTF)

clean:
	rm *.o $(OUTF)
