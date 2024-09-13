class WordCounter
	attr_reader :counts

	def process(file_content)
		@counts = Hash.new(0)
		
		file_content.each do |line|
			line.split.each do |word|
				@counts[word.downcase.gsub(/\W/, '')] += 1
			end
		end
	end

	def print
		@counts.sort_by { |word, count| -count }.each do |word, count|
			puts "#{word}: #{count}"
		end
	end
end

content = File.read(File.expand_path("../shakespeare.txt", __FILE__))

counter = WordCounter.new
counter.process(content)
counter.print