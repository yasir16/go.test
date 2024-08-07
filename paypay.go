package main

import (
	"fmt"
	"sort"
	"strings"
)

func main() {
	data := []string{
		"APPX,170.00",
		"AMZY,150.00",
		"APPX,180.00",
		"APPX,165.00",
		"APPX,185.00",
		"AMZY,145.00",
	}

	fmt.Println(permuntation(10))
	fmt.Println(solution(data))
}
func solution(csv []string) []string {
	resultMap := make(map[string]*result)
	var resu []string
	for _, row := range csv {
		dataRow := strings.Split(row, ",")
		// tmpPrice, _ := strconv.ParseFloat(dataRow[1], 64)
		if resultMap[dataRow[0]] != nil {
			// dataHighPrice, _ := strconv.ParseFloat(resultMap[dataRow[0]].highPrice, 64)
			// dataLowPrice, _ := strconv.ParseFloat(resultMap[dataRow[0]].lowPrice, 64)

			if dataRow[1] > resultMap[dataRow[0]].highPrice {
				resultMap[dataRow[0]].highPrice = dataRow[1]
			}
			if dataRow[1] < resultMap[dataRow[0]].lowPrice {
				resultMap[dataRow[0]].lowPrice = dataRow[1]
			}
			resultMap[dataRow[0]].closePrice = dataRow[1]
		} else {
			resultMap[dataRow[0]] = &result{
				brandID:    dataRow[0],
				highPrice:  dataRow[1],
				openPrice:  dataRow[1],
				closePrice: dataRow[1],
				lowPrice:   dataRow[1],
				// highPrice:  fmt.Sprintf("%.2f", tmpPrice),
				// openPrice:  fmt.Sprintf("%.2f", tmpPrice),
				// closePrice: fmt.Sprintf("%.2f", tmpPrice),
				// lowPrice:   fmt.Sprintf("%.2f", tmpPrice),
			}
		}
	}

	for _, res := range resultMap {
		tmp := fmt.Sprintf("%s,%s,%s,%s,%s", res.brandID, res.highPrice, res.openPrice, res.closePrice, res.lowPrice)
		resu = append(resu, tmp)
	}
	sort.Strings(resu)
	return resu
}

type result struct {
	brandID    string
	highPrice  string
	openPrice  string
	closePrice string
	lowPrice   string
}

func factorial(n int) uint64 {
	if n == 1 {
		return 1
	}
	return uint64(n) * factorial(n-1)
}

func permuntation(n int) uint64 {
	return factorial(n) / factorial(n-1)
}
