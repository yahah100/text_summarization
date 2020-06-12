package main

import (
	"bufio"
	"fmt"
	"github.com/abadojack/whatlanggo"
	"log"
	"os"
	"regexp"
	"strings"
)

func check(e error) {
	if e != nil {
		panic(e)
	}
}

func main() {

	file_de, err := os.Open("data/train.de")
	check(err)
	defer file_de.Close()

	file_en, err := os.Open("data/train.en")
	check(err)
	defer file_de.Close()

	scanner_de := bufio.NewScanner(file_de)
	scanner_en := bufio.NewScanner(file_en)

	file_out_de, err := os.OpenFile("data/train_1.de", os.O_APPEND|os.O_WRONLY, 0644)
	check(err)
	defer file_out_de.Close()

	file_out_en, err := os.OpenFile("data/train_1.en", os.O_APPEND|os.O_WRONLY, 0644)
	check(err)
	defer file_out_de.Close()

	count := 0
	for scanner_de.Scan() && scanner_en.Scan(){
		line_de := scanner_de.Text()
		line_en := scanner_en.Text()

		line_de = strings.Replace(line_de, "##AT##-##AT## ", "", -1)
		line_en = strings.Replace(line_en, "##AT##-##AT## ", "", -1)

		r := regexp.MustCompile("[^a-zA-Z0-9üöä \n.()#-:]")
		line_de = r.ReplaceAllString(line_de, "")
		line_en = r.ReplaceAllString(line_en, "")

		info := whatlanggo.Detect(line_de)
		lang := info.Lang.String()

		if lang == "German" {
			_, err = fmt.Fprintln(file_out_de, line_de)
			_, err = fmt.Fprintln(file_out_en, line_en)
		}

		count += 1
	}

	if err := scanner_de.Err(); err != nil {
		log.Fatal(err)
	}
	if err := scanner_en.Err(); err != nil {
		log.Fatal(err)
	}
}