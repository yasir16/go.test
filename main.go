package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

// output
//1624507883, JOHN DOE, DEBIT, 250000, SUCCESS, restaurant
//1624608050, E-COMMERCE A, DEBIT, 150000, FAILED, clothes
//1624512883, COMPANY A, CREDIT, 12000000, SUCCESS, salary
//1624615065, E-COMMERCE B, DEBIT, 150000, PENDING, clothes

type transaction struct {
	transactionID string
	name          string
	source        string
	amounth       float32
	status        string
	detailTrx     string
}

func read(filename string) []transaction {

	file, err := os.Open(filename)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		panic(err)
	}

	var transactions []transaction

	test := make(map[string]*transaction)
	if test["sa"] == nil {

	}

	for idx, rec := range records {
		if idx == 0 {
			continue
		}
		amountStr := strings.Trim(rec[3], " ")
		amount, err := strconv.ParseFloat(amountStr, 32)
		if err != nil {
			panic(err)
		}
		transactions = append(transactions, transaction{
			transactionID: strings.Trim(rec[0], " "),
			name:          strings.Trim(rec[1], " "),
			source:        strings.Trim(rec[2], " "),
			amounth:       float32(amount),
			status:        strings.Trim(rec[4], " "),
			detailTrx:     strings.Trim(rec[5], " "),
		})
	}

	return transactions
}

func main2() {
	datas := read("data.csv")
	fmt.Printf("Balance : %.2f\n", calculateBallance(datas))

	var root *TreeNode
	// [1,2,3,null,5,null,4]
	root = &TreeNode{
		Val: 1,
		Left: &TreeNode{
			Val: 2,
			Right: &TreeNode{
				Val: 5,
			},
		},
		Right: &TreeNode{
			Val: 3,
			Right: &TreeNode{
				Val: 4,
			},
		},
	}

	binStr := fmt.Sprintf("%b", 561892)

	tmpGap, maxGap := 0, 0

	for _, str := range binStr {
		if str == rune('0') {
			tmpGap++
		} else if str == rune('1') && tmpGap == 0 {
			continue
		} else if str == rune('1') && tmpGap > 0 {
			if maxGap < tmpGap {
				maxGap = tmpGap
			}
			tmpGap = 0
		}
	}
	fmt.Println("max gap : ", maxGap)

	fmt.Println(rightSideView(root))

	// fmt.Println(letterCombination("23"))

	// fmt.Println(groupAnagrams([]string{"eat", "tea", "tan", "ate", "nat", "bat"}))

	// fmt.Println(findSubstring("barfoothefoobarman", []string{"foo", "bar"}))
	fmt.Println(convert("AB", 1))

	fmt.Println("max sliding window: ", maxSlidingWindow([]int{7, 2, 4}, 2))

	fmt.Println("max sliding window satu: ", maxSlidingWindowSatu([]int{7, 2, 4}, 2))

	fmt.Println("max sliding window dua: ", maxSlidingWindowDua([]int{7, 2, 4}, 2))

	fmt.Println(isPalindrome("fdasf"))
	fmt.Println(isValid("{())}"))

	p := 34
	q := 20
	bitP := strconv.FormatInt(int64(p), 2)
	bitQ := strconv.FormatInt(int64(q), 2)
	fmt.Println("P : ", bitP)
	fmt.Println("Q : ", bitQ)

	// & (bitwise AND)
	result1 := p & q
	fmt.Printf("Result of p & q = %d", result1)

	// | (bitwise OR)
	result2 := p | q
	fmt.Printf("\nResult of p | q = %d", result2)

	// ^ (bitwise XOR)
	result3 := p ^ q
	fmt.Printf("\nResult of p ^ q = %d", result3)

	// << (left shift)
	result4 := p << 1
	fmt.Printf("\nResult of p << 1 = %d", result4)

	// >> (right shift)
	result5 := p >> 1
	fmt.Printf("\nResult of p >> 1 = %d", result5)

	// &^ (AND NOT)
	result6 := p &^ q
	fmt.Printf("\nResult of p &^ q = %d\n", result6)

	fmt.Println("^p: ", ^p)
	fmt.Println(strconv.FormatInt(int64(^p), 2))
	fmt.Println("^q: ", ^q)
	fmt.Println(strconv.FormatInt(int64(^q), 2))
	fmt.Println(minWindow("ADOBECODEBANC", "ABC"))
	fmt.Println(minWindowEasy("ADOBECODEBANC", "ABC"))
	fmt.Println(calCalculator(" 3+5 / 2 "))
	s := "   fly me   to   the moon  "
	fmt.Println(lengthOfLastWord(s))

	fmt.Println(longestPalindrome("babad"))

	fmt.Println(isHappy(123))

	fmt.Println(centuryYear(1908))
	fmt.Println(minSubArrayLen(7, []int{2, 3, 1, 2, 4, 3}))

}

func calculateBallance(datas []transaction) float32 {
	var ballance float32
	for _, data := range datas {
		if data.status == "SUCCESS" {
			if data.source == "DEBIT" {
				ballance += data.amounth
			} else if data.source == "CREDIT" {
				ballance -= data.amounth
			}
		}
	}
	return ballance
}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func rightSideView(root *TreeNode) []int {
	if root == nil {
		return []int{}
	}
	var res []int
	queue := []*TreeNode{root}
	for len(queue) > 0 {
		nodesInCurrentLevel := len(queue)
		prev := 0
		for i := 0; i < nodesInCurrentLevel; i++ {
			node := queue[0]
			queue = queue[1:]
			prev = node.Val
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {

				queue = append(queue, node.Right)
			}
		}
		res = append(res, prev)
	}
	return res

}

func hash(s string) string {
	res := make([]byte, 26)
	for _, c := range s {
		res[c-'a'] += 1
	}
	return string(res)
}

func groupAnagrams(strs []string) [][]string {
	res := [][]string{}
	m := make(map[string]int)
	for _, w := range strs {
		h := hash(w)
		idx, ok := m[h]
		if ok {
			res[idx] = append(res[idx], w)
		} else {
			res = append(res, []string{w})
			m[h] = len(res) - 1
		}
	}

	return res
}

func letterCombination(digit string) []string {
	if digit == "" {
		return []string{}
	}
	var output []string
	phoneKeyPadMap := []string{"abc",
		"def",
		"ghi",
		"jkl",
		"mno",
		"pqrs",
		"tuv",
		"wxyz",
	}
	var backtrack func(combination, nextDigit string)
	backtrack = func(combination, nextDigit string) {
		if nextDigit == "" {
			output = append(output, combination)
		} else {
			letters := phoneKeyPadMap[nextDigit[0]-'2']
			for _, letter := range letters {
				backtrack(combination+string(letter), nextDigit[1:])
			}
		}
	}
	backtrack("", digit)

	return output
}

func findSubstring(s string, words []string) []int {
	wordLength := len(words[0]) // They all have the same length
	concatenatedSubstringLength := wordLength * len(words)
	if concatenatedSubstringLength > len(s) {
		return []int{}
	}

	startingIndexes := []int{}
	for index := 0; index <= len(s)-concatenatedSubstringLength; index++ {
		remainingWords := make([]string, len(words))
		_ = copy(remainingWords, words)
		// At the index, check whether all words are present
		for wordIndex := index; len(remainingWords) > 0; wordIndex += wordLength {
			wordAtIndex := s[wordIndex : wordIndex+wordLength]
			found := false
			for i, w := range remainingWords {
				if wordAtIndex == w {
					found = true
					// Remove the element
					remainingWords[i] = remainingWords[len(remainingWords)-1]
					remainingWords = remainingWords[:len(remainingWords)-1]
					break
				}
			}
			if !found {
				break
			}
		}
		// If all words are present, append this index to the solution slice
		if len(remainingWords) == 0 {
			startingIndexes = append(startingIndexes, index)
		}
	}

	return startingIndexes
}

func convert(s string, numRows int) string {
	if numRows == 1 {
		return s
	}
	tmpArr := make([]string, numRows)
	counter, step := 0, 1

	for _, str := range s {
		tmpArr[counter] += string(str)

		if counter == 0 {
			step = 1
		} else if counter == numRows-1 {
			step = -1
		}
		counter += step

	}

	return strings.Join(tmpArr, "")
}

func maxSlidingWindow(nums []int, k int) []int {
	var res []int

	for i := 0; i <= len(nums)-k; i++ {
		tmpArr := make([]int, k)
		_ = copy(tmpArr, nums[i:i+k])
		sort.Ints(tmpArr)
		res = append(res, tmpArr[len(tmpArr)-1])
	}
	return res
}

func maxSlidingWindowSatu(nums []int, k int) []int {
	// Start with a window of length zero
	left, right := 0, 0

	// We will use this array as a deque ( a queue that allows both front & back pop)
	// This is a monotonic increasing deque, meaning smaller elements will be popped
	// This is similar to monotonic increasing stack, but can do front pop
	deque := []int{}

	res := []int{}
	for right < len(nums) {
		// If there are elements on top of this deque smaller than current element
		for len(deque) > 0 && nums[right] > nums[deque[len(deque)-1]] {
			// Pop the smaller element
			deque = deque[:len(deque)-1]
		}

		// Append the new element's index
		// We are appending index and not the element itself cause
		// With Index we can make judgement about our window length
		deque = append(deque, right)

		// If left has moved beyond the index of first element, pop
		if left > deque[0] {
			deque = deque[1:]
		}

		// Check if we have a valid window
		if right+1 >= k {
			// Append the first number from deque to the result
			// The first number will always be the largest
			res = append(res, nums[deque[0]])
			left++
		}
		right++
	}

	return res
}

func maxSlidingWindowDua(nums []int, k int) []int {
	leftArr := []int{}
	rightArr := make([]int, len(nums))
	leftMax := 0
	rightMax := 0
	result := make([]int, len(nums)-k+1)
	for i, n := range nums {
		if i%k == 0 {
			leftMax = math.MinInt
		}
		leftMax = max(leftMax, n)
		leftArr = append(leftArr, leftMax)
	}
	for i := len(nums) - 1; i >= 0; i-- {
		if i%k == 0 {
			rightMax = math.MinInt
		}
		rightMax = max(rightMax, nums[i])
		rightArr[i] = rightMax
	}

	for i := 0; i < len(result); i++ {
		result[i] = max(rightArr[i], leftArr[i+k-1])
	}

	return result
}

func isPalindrome(s string) bool {
	left, right := 0, len(s)-1
	for left < right {
		l := rune(s[left])
		r := rune(s[right])

		if !unicode.IsLetter(l) && !unicode.IsDigit(l) {
			left++
		} else if !unicode.IsLetter(r) && !unicode.IsDigit(r) {
			right--
		} else if unicode.ToLower(l) == unicode.ToLower(r) {
			left++
			right--
		} else {
			return false
		}
	}
	return true

}

// {())}
func isValid(s string) bool {
	stack := []rune{}

	bracketMap := map[rune]rune{
		')': '(',
		'}': '{',
		']': '[',
	}

	for _, char := range s {
		if matchingBracket, found := bracketMap[char]; found {
			if len(stack) > 0 && stack[len(stack)-1] == matchingBracket {
				stack = stack[:len(stack)-1]
			} else {
				return false
			}
		} else {
			stack = append(stack, char)
		}
	}

	return len(stack) == 0
}

func minWindow(s string, t string) string {
	if len(s) == 0 || len(t) == 0 || len(s) < len(t) {
		return ""
	}

	mapS := make([]int, 128)
	count := len(t)
	start, end := 0, 0
	minLen, startIndex := int(^uint(0)>>1), 0
	/// UPVOTE !
	for _, char := range t {
		mapS[char]++
	}

	for end < len(s) {
		if mapS[s[end]] > 0 {
			count--
		}
		mapS[s[end]]--
		end++

		for count == 0 {
			if end-start < minLen {
				startIndex = start
				minLen = end - start
			}

			if mapS[s[start]] == 0 {
				count++
			}
			mapS[s[start]]++
			start++
		}
	}

	if minLen == int(^uint(0)>>1) {
		return ""
	}

	return s[startIndex : startIndex+minLen]

}

func minWindowEasy(s string, t string) string {
	answer := ""
	if len(t) > len(s) {
		return answer
	}
	n := len(t)
	m := len(s)
	targetMap := make(map[rune]int, n)
	tRune := []rune(t)
	sRune := []rune(s)
	for _, r := range tRune {
		count, _ := targetMap[r]
		targetMap[r] = count + 1
	}
	matchCount := 0
	start := 0
	minLength := m + 2
	min_start := 0
	for i, r := range sRune {
		if count, ok := targetMap[r]; ok {
			if count > 0 {
				matchCount++
			}
			targetMap[r] = count - 1
		}
		if matchCount == n {
			tempCount, ok := targetMap[sRune[start]]
			for (start < i && !ok) || (ok && tempCount < 0) {
				if ok {
					targetMap[sRune[start]] = tempCount + 1
				}
				start++
				tempCount, ok = targetMap[sRune[start]]
			}
			if minLength > i-start+1 {
				answer = string(sRune[start : i+1])
				min_start = start
				minLength = i - start + 1
			}
		}
	}
	if minLength == m+2 {
		return ""
	}
	answer = string(sRune[min_start : min_start+minLength])
	return answer
}

func calCalculator(s string) int {
	curr := 0
	prev := 0
	res := 0
	op := '+'

	for i, ch := range s {
		if unicode.IsDigit(ch) {
			tmp := int(ch - '0')
			curr = curr*10 + tmp
		}

		if !unicode.IsSpace(ch) && !unicode.IsDigit(ch) || i == len(s)-1 {
			switch op {
			case '+':
				res += prev
				prev = curr
			case '-':
				res += prev
				prev = -curr
			case '*':
				prev *= curr
			case '/':
				prev /= curr
			}
			op = ch
			curr = 0
		}
	}

	res += prev
	return res
}

func lengthOfLastWord(s string) int {
	arrStr := strings.Split(s, " ")
	res := arrStr[len(arrStr)-1]
	if res == " " {
		for i := len(arrStr) - 2; i >= 0; i-- {
			fmt.Println(arrStr[i])
			if arrStr[i] != "" {
				res = arrStr[i]
				break
			}
		}

	}
	return len(res)
}

func longestPalindrome(s string) string {
	len := len(s)
	if len == 0 {
		return ""
	}

	var l, r, pl, pr int
	for r < len {
		// gobble up dup chars
		for r+1 < len && s[l] == s[r+1] {
			r++
		}
		// find size of this palindrome
		for l-1 >= 0 && r+1 < len && s[l-1] == s[r+1] {
			l--
			r++
		}
		if r-l > pr-pl {
			pl, pr = l, r
		}
		// reset to next mid point
		l = (l+r)/2 + 1
		r = l
	}
	return s[pl : pr+1]
}

func isHappy(n int) bool {
	var crap int
	for n > 0 {
		tmp := n % 10
		crap += tmp * tmp
		n /= 10
	}

	if crap == 1 {
		return true
	} else if crap == 4 {
		return false
	} else {
		return isHappy(crap)
	}

}

func centuryYear(year int) int {
	if year <= 0 {
		return 1
	}
	if year%100 > 0 {
		return (year / 100) + 1
	}
	return year / 100
}

func minSubArrayLen(target int, nums []int) int {
	out := 0

	start, sum := 0, 0

	for i := 0; i < len(nums); i++ {
		sum += nums[i]

		for sum >= target {
			if out == 0 || (i-start+1) < out {
				out = i - start + 1
			}

			sum -= nums[start]
			start++
		}
	}
	return out

}
