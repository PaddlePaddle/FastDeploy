.PHONY: clean all test doc lint show_docs

all: bin/mock

LDFLAGS =
CFLAGS = -Wall -O3 -Iinclude -Icontrib
CXXFLAGS = -std=c++11 $(CFLAGS)

SRC = $(wildcard contrib/*.cc contrib/*.c)
ALL_CXX_OBJ = $(patsubst contrib/%.cc, build/%.o, $(SRC))
ALL_C_OBJ = $(patsubst contrib/%.c, build/%.o, $(SRC))
ALL_OBJ = $(ALL_CC_OBJ) $(ALL_CXX_OBJ)

doc:
	doxygen docs/Doxyfile
	$(MAKE) -C docs html

show_docs:
	@python3 -m http.server --directory docs/build/latest

lint:
	./tests/scripts/task_lint.sh

build/%.o: contrib/%.cc
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -MM -MT build/$*.o $< >build/$*.d
	$(CXX) -c $(CXXFLAGS) -c $< -o $@

build/%.o: contrib/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -MM -MT build/$*.o $< >build/$*.d
	$(CC) -c $(CFLAGS) -c $< -o $@

bin/mock: $(ALL_OBJ)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $@ $(filter %.o %.a, $^) $(LDFLAGS)

clean:
	$(RM) -rf build  */*/*/*~ */*.o */*/*.o */*/*/*.o */*.d */*/*.d */*/*/*.d docs/build docs/doxygen
