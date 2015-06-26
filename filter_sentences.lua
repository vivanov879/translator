require 'mobdebug'.start()
require 'table_utils'

fd = io.lines('en')
words_count = {}
words = {}
sentences = {}
vocab_size = 10000
line = fd()
while line do
  sentence = {}
  for _, word in pairs(string.split(line, " ")) do
    if word ~= '.' 
      and word ~= ',' 
      and word ~= '&quot;' 
      and word ~= '(' 
      and word ~= ')' 
      and word ~= '-' 
      and word ~= ";" 
      and word ~= ":" 
      and word ~= '..' 
      and word ~= '...' 
      and word ~= '/' 
      and word ~= '\\' 
      and word ~= '?' 
      and word ~= '»' 
      and word ~= '«'
      and word ~= '—'
      and word ~= '&#91'
      and word ~= '&gt;'
      and word ~= '&apos;s'
      and word ~= '&apos;t'
      and word ~= '&apos;re'
      and word ~= '&apos;ll'
      and word ~= '&apos;'
      and word ~= '&apos;'
      and word ~= '&apos;'
      and word ~= '&lt;'
        then
      sentence[#sentence + 1] = word
    end
  end
  sentences[#sentences + 1] = sentence
  
  for _, word in pairs(sentence) do
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
--for key, val in pairs(words_count) do  -- Table iteration.
--  print(key, val)
--end

vocabulary = {}
inv_vocabulary = {}
for i = 1, vocab_size do 
  vocabulary[i] = words[i]
  inv_vocabulary[words[i]] = i
end
vocabulary[#vocabulary + 1] = 'UNK'
inv_vocabulary['UNK'] = #vocabulary
vocabulary[#vocabulary + 1] = 'EOS'
inv_vocabulary['EOS'] = #vocabulary

--print (vocabulary)
--print(inv_vocabulary)
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
    else
      filtered_sentence[#filtered_sentence + 1] = 'UNK'
    end
  end
  --filtered_sentence = table.reverse(filtered_sentence)
  filtered_sentence[#filtered_sentence + 1] = 'EOS'
  --print(#filtered_sentence, #sentence)
  filtered_sentences[#filtered_sentences + 1] = filtered_sentence
  
end

--print(filtered_sentences)

sentence_lengths = torch.zeros(#sentences)
for i = 1, #sentences do
  sentence_lengths[i] = #(sentences[i])
end

filtered_sentence_lengths = torch.zeros(#filtered_sentences)
for i = 1, #filtered_sentences do
  filtered_sentence_lengths[i] = #(filtered_sentences[i])
end

filtered_sentences_indexes = {}
for _, filtered_sentence in pairs(filtered_sentences) do
  sentence = {}
  for _, word in pairs(filtered_sentence) do  
    sentence[#sentence + 1] = inv_vocabulary[word]
  end
  filtered_sentences_indexes[#filtered_sentences_indexes + 1] = sentence
end

--print(filtered_sentences_indexes)


print(torch.mean(sentence_lengths), torch.std(sentence_lengths), torch.max(sentence_lengths))
print(torch.mean(filtered_sentence_lengths), torch.std(filtered_sentence_lengths), torch.min(filtered_sentence_lengths), torch.max(filtered_sentence_lengths))

print(#filtered_sentences)


fd = io.open('filtered_sentences_en', 'w')
for _, sentence in pairs(filtered_sentences) do
  fd:write(table.concat(sentence, ' ') .. '\n')
end

fd = io.open('filtered_sentences_indexes_en', 'w')
for _, sentence in pairs(filtered_sentences_indexes) do
  fd:write(table.concat(sentence, ' ')  .. '\n')
end

a = 1


