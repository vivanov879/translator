require 'mobdebug'.start()
require 'table_utils'


words2omit = {["!"] = 177, ["#"] = 2930, ["$"] = 314, ["%"] = 139, ["&#91;"] = 312, ["&#93;"] = 316, ["&amp;"] = 482, ["&apos;"] = 92, ["&apos;an"] = 9189, ["&apos;d"] = 1069, ["&apos;i"] = 5286, ["&apos;ll"] = 566, ["&apos;m"] = 808, ["&apos;re"] = 358, ["&apos;s"] = 26, ["&apos;t"] = 85, ["&apos;ve"] = 555, ["&gt;"] = 730, ["&lt;"] = 2691, ["&quot;"] = 12, ["("] = 35, [")"] = 33, ["*"] = 339, ["+"] = 795, [","]
= 3, ["-"] = 54, ["--"] = 716, ["."] = 2, [".."] = 2947, ["..."] = 247, ["...."] = 4014, [".net"] = 3428, ["/"] = 75, ["0"] = 2316, ["000"] = 4396, ["00"] = 2005, ["0.5"] = 7859, ["01"] = 6862, ["1"] = 222, ["1,000"] = 4242, ["1."] = 1300, ["1.0"] = 7528, ["1.1"] = 6694, ["1.2"] = 6832, ["1.3"] = 8018, ["1.4"] = 9711, ["1.5"] = 4156, ["1.6"] = 8774, ["1.8"] = 9963, ["1st"] = 3930, ["2"] = 238, ["2,000"]
= 6466, ["2-3"] = 6955, ["2-minute"] = 9010, ["2."] = 1229, ["2.0"] = 4407, ["2.1"] = 9735, ["2.2"] = 7247, ["2.5"] = 5633, ["2.8"] = 7673, ["2nd"] = 6812, ["3"] = 356, ["3,000"] = 7919, ["3-4"] = 8690, ["3."] = 1400, ["3.0"] = 5553, ["3.1"] = 9867, ["3.2"] = 9107, ["3.5"] = 5586, ["3.6"] = 9320, ["3d"] = 3708, ["3g"] = 9218, ["3rd"] = 6331, ["4"] = 474, ["4,000"] = 8857, ["4."] = 1881, ["4.0"] =
6230, ["4.2"] = 8779, ["4.3"] = 9825, ["4.5"] = 7668, ["4th"] = 6710, ["5"] = 400, ["5-6"] = 9930, ["5-minute"] = 5058, ["5."] = 2616, ["5.5"] = 8908, ["5th"] = 6887, ["6"] = 779, ["6."] = 2808, ["6.0"] = 8951, ["6th"] = 6605, ["7"] = 811, ["07"] = 7037, ["7,000"] = 9182, ["7."] = 3306, ["7th"] = 8414, ["8"] = 940, ["08"] = 8865, ["8."] = 4133, ["09"] = 9400, ["9"] = 1211, ["9."] = 4395, ["9th"] =
9808, ["10"] = 311, ["10,000"] = 4589, ["10-minute"] = 4643, ["10."] = 4621, ["10th"] = 7888, ["11"] = 1051, ["11."] = 6319, ["11th"] = 7332, ["12"] = 718, ["12."] = 6740, ["12th"] = 8203, ["13"] = 1338, ["13."] = 7729, ["14"] = 1368, ["14."] = 6705, ["15"] = 617, ["15,000"] = 8262, ["15-minute"] = 6130, ["15."] = 8488, ["15th"] = 7537, ["16"] = 1286, ["16."] = 7187, ["16th"] = 7028, ["17"] = 1458,
["17."] = 8341, ["17th"] = 6554, ["18"] = 1194, ["18."] = 7736, ["18th"] = 7469, ["19"] = 1847, ["19th"] = 4889, ["20"] = 519, ["20,000"] = 7848, ["20-minute"] = 8861, ["20."] = 9216, ["20th"] = 4055, ["21"] = 1777, ["21."] = 9414, ["21st"] = 6058, ["22"] = 2032, ["22."] = 9936, ["23"] = 2321, ["24"] = 1472, ["24-hour"] = 5209, ["25"] = 1254, ["26"] = 2305, ["27"] = 2377, ["28"] = 2332, ["29"] = 2638,
["30"] = 582, ["30th"] = 9768, ["31"] = 2987, ["32"] = 3178, ["32-bit"] = 7819, ["33"] = 4799, ["34"] = 5494, ["35"] = 2845, ["36"] = 5288, ["37"] = 4662, ["38"] = 5690, ["39"] = 5438, ["40"] = 1175, ["41"] = 6558, ["42"] = 4145, ["42."] = 9697, ["43"] = 8376, ["44"] = 5540, ["45"] = 3855, ["46"] = 7054, ["47"] = 5900, ["48"] = 4714, ["49"] = 5490, ["50"] = 996, ["50,000"] = 7222, ["51"] = 7446, ["52"]
= 6519, ["53"] = 7133, ["54"] = 9291, ["55"] = 5993, ["56"] = 5833, ["57"] = 6381, ["58"] = 6779, ["59"] = 8356, ["60"] = 1822, ["60th"] = 9997, ["61"] = 9440, ["62"] = 6781, ["63"] = 9032, ["64"] = 5984, ["64-bit"] = 7596, ["65"] = 5614, ["66"] = 7568, ["68"] = 6129, ["69"] = 9519, ["70"] = 2542, ["72"] = 8836, ["75"] = 4078, ["76"] = 8804, ["78"] = 7972, ["80"] = 1894, ["83"] = 7869, ["85"] = 6894,
["90"] = 2527, ["92"] = 9050, ["94"] = 8565, ["95"] = 4990, ["96"] = 8489, ["98"] = 5281, ["99"] = 8778, ["100"] = 757, ["100,000"] = 6335, ["110"] = 7532, ["120"] = 4256, ["130"] = 9678, ["140"] = 6998, ["150"] = 3456, ["160"] = 8775, ["180"] = 8084, ["200"] = 1953, ["250"] = 4376, ["300"] = 2809, ["360"] = 8207, ["365"] = 9283, ["400"] = 3071, ["450"] = 7880, ["500"] = 1989, ["500,000"] = 9346,
["600"] = 4852, ["700"] = 5512, ["800"] = 4509, ["900"] = 7214, ["1000"] = 3969, ["1200"] = 9407, ["1500"] = 8003, ["1900"] = 9699, ["1917"] = 8069, ["1918"] = 8430, ["1919"] = 9981, ["1920"] = 7796, ["1922"] = 9756, ["1925"] = 8751, ["1929"] = 8835, ["1930s"] = 8611, ["1937"] = 8904, ["1940"] = 7676, ["1941"] = 8034, ["1943"] = 8274, ["1944"] = 6502, ["1945"] = 6637, ["1946"] = 9096, ["1948"] = 6881,
["1949"] = 8843, ["1950"] = 6991, ["1950s"] = 8124, ["1952"] = 9774, ["1953"] = 8193, ["1954"] = 9574, ["1955"] = 8319, ["1956"] = 8504, ["1957"] = 8701, ["1958"] = 8024, ["1959"] = 8628, ["1960"] = 7095, ["1960s"] = 7775, ["1964"] = 9215, ["1965"] = 9921, ["1967"] = 7714, ["1968"] = 7883, ["1969"] = 9575, ["1970"] = 6572, ["1970s"] = 5587, ["1971"] = 8164, ["1972"] = 6125, ["1973"] = 5938, ["1974"]
= 8507, ["1975"] = 7589, ["1976"] = 7760, ["1977"] = 7376, ["1978"] = 6032, ["1979"] = 5246, ["1980"] = 6387, ["1980s"] = 5228, ["1981"] = 8072, ["1982"] = 7650, ["1983"] = 8897, ["1984"] = 6612, ["1985"] = 5899, ["1986"] = 5889, ["1987"] = 7971, ["1988"] = 4274, ["1989"] = 4890, ["1990"] = 3550, ["1990s"] = 3792, ["1991"] = 2971, ["1992"] = 3037, ["1993"] = 3648, ["1994"] = 3106, ["1995"] = 2850,
["1996"] = 2459, ["1997"] = 2793, ["1998"] = 2204, ["1999"] = 1714, ["2000"] = 978, ["2001"] = 1475, ["2002"] = 1406, ["2003"] = 698, ["2004"] = 1020, ["2005"] = 761, ["2006"] = 688, ["2007"] = 398, ["2008"] = 411, ["2009"] = 585, ["2010"] = 686, ["2011"] = 1501, ["2012"] = 1643, ["2013"] = 3186, ["2014"] = 8081, ["2015"] = 7465, ["2020"] = 8777, ["3000"] = 7536, ["5000"] = 9966, [":"] = 36, [";"]
= 90, ["="] = 3533, ["?"] = 65, ["@"] = 4342, ["\\"] = 1674, ["^"] = 9971, _ = 512, ["1,500"] = 9840, ["1-2"] = 9847, ["1-extra"] = 9841, ["2.7"] = 9666, ["4x4"] = 9962, ["7.5"] = 9875, ["16-bit"] = 9995, ["256"] = 9759, ["1913"] = 9898, ["1930"] = 9988, ["1933"] = 9719, ["1936"] = 9758, ["1939"] = 9872, ["1942"] = 9845, ["4000"] = 2233, ["{"] = 9277, ["}"] = 8646, ["~"] = 7491, ["§"] = 5832, ["«"] = 1895, ["®"] = 3213, ["°"] = 4339, ["·"] = 4518, ["»"] = 1750, ["à"] = 8077, ["ñ"] = 7898, ["–"] = 359, ["—"] = 1137, ["‘"] = 2386, ["’"] = 114, ["“"] = 219, ["”"] = 205, ["•"] = 1931, ["…"] = 1921, ["€"] = 2759, ["№"] = 4827, ["™"] = 2274, ["→"] = 7895 }
function word2omit(word)
  for key, _ in pairs(words2omit) do
    if word == key then 
      return true
    end
    
  end
end

vocabulary_fn = 'vocabulary_en'
inv_vocabulary_fn = 'inv_vocabulary_en'
filtered_sentences_fn = 'filtered_sentences_en'
filtered_sentences_indexes_fn = 'filtered_sentences_indexes_en'
should_reverse = false

fd = io.lines('en')
words_count = {}
words = {}
sentences = {}
vocab_size = 10000
line = fd()
while line do
  sentence = {}
  for _, word in pairs(string.split(line, " ")) do
    if not word2omit(word) then
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

table.save(vocabulary, vocabulary_fn)
table.save(inv_vocabulary, inv_vocabulary_fn)

print (vocabulary)
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
  if should_reverse then 
    filtered_sentence = table.reverse(filtered_sentence)
  end
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


fd = io.open(filtered_sentences_fn, 'w')
for _, sentence in pairs(filtered_sentences) do
  fd:write(table.concat(sentence, ' ') .. '\n')
end

fd = io.open(filtered_sentences_indexes_fn, 'w')
for _, sentence in pairs(filtered_sentences_indexes) do
  fd:write(table.concat(sentence, ' ')  .. '\n')
end

a = 1


