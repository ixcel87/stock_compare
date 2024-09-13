words = {}

File.open(File.expand_path("../shakespeare.txt", __FILE__), 'r').each do |line|
	line.split.each do |word|
		clean_word = word.downcase.gsub(/\W/, '')
		
		word_count = words[clean_word]
		word_count = 0 if word_count == nil
		word_count += 1
		
		words[clean_word] = word_count
	end
end

words = words.sort_by { |key, value| value }

words.reverse.each do |key, value|
	puts "#{key}: #{value}"
end