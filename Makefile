#Utilize esse arquivo para compilar o projeto como um todo para sua máquina

all: executavel

.PHONY: executavel 


executavel: src/main.cpp 
	g++ $^ -o $@ 
