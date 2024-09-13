# encoding: utf-8

class Mimic
	def initialize
		@words = Hash.new { |hash, key| hash[key] = [] } # exec every time missing key is looked up
	end

	def process(input)
		line_words = input.split
		line_words.each_with_index do |word, index|
			@words[word] << line_words[index + 1]
		end
	end

	def print(max_length, start_word)
		if !@words.has_key?(start_word)
			puts 'Start word does not match any processed words!'
			return
		end

		sentence = start_word
		current_word = start_word
		until sentence.length >= max_length do
			next_word = random_possible_word_for(current_word)
			break if next_word == nil

			sentence += ' ' + next_word
			current_word = next_word
		end

		puts sentence
	end

	private

	def random_possible_word_for(word)
		@words[word].choice
	end
end

m = Mimic.new
m.process(File.read(File.expand_path("../alice_in_wonderland.txt", __FILE__)))
m.print(200, 'Alice')