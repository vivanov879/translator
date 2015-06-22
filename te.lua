require 'mobdebug'.start()





fd = io.lines('en')
words_count = {}
words = {}
sentences = {}
line = fd()
while line do
  sentence = {}
  for word in string.gmatch(line, "%a+") do
    sentence[#sentence + 1] = word
  end
  sentences[#sentences + 1] = sentence
  
  for word in string.gmatch(line, "%a+") do
    if words_count[word] then
      words_count[word] = words_count[word] + 1
    else
      words_count[word] = 1
      words[#words + 1] = word
    end
    --print(word)
  end
  line = fd()
end


function compare(a, b)    
    if words_count[a] > words_count[b] then    
        return true    
    end
end

table.sort(words, compare)
print(words)
--for key, val in pairs(words_count) do  -- Table iteration.
--  print(key, val)
--end

vocabulary = {}
for i = 1, 300 do 
  vocabulary[#vocabulary + 1] = words[i]
end

function in_array(x, l)
  for i = 1, #l do
    if l[i] == x then
      return true
    end
  end
end
  
empty_sentence_indexes = {}

filtered_sentences = {}
for i = 1, #sentences do
  sentence = sentences[i]
  filtered_sentence = {}
  for k = 1, #sentence do
    word = sentence[k]
    if in_array(word, vocabulary) then
      filtered_sentence[#filtered_sentence + 1] = word
    end
  end
  --print(#filtered_sentence, #sentence)
  if #filtered_sentence == 0 then
    empty_sentence_indexes[#empty_sentence_indexes + 1] = i
    --print(sentence)
  end
  filtered_sentences[#filtered_sentences + 1] = filtered_sentence
end

print(#empty_sentence_indexes, #sentences)

sentence_lengths = torch.zeros(#sentences)
for i = 1, #sentences do
  sentence_lengths[i] = #(sentences[i])
end

print(torch.mean(sentence_lengths), torch.std(sentence_lengths))


a = 1


