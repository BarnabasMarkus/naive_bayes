# naive_bayes
Implementation of Naive-Bayes Classifier

## Motivation
This classifier is a part of a chatbot's text classifier.
I have started the development of my own Naive-Bayes Classifier implementation in order to keep it as simple and customizable as possible.

## Usage

```Python
>>> 
>>> from naive_bayes import NaiveBayes
>>> 
>>> bayes = NaiveBayes()
>>> 
>>> # Start train the classifier
>>> bayes.train('free poker cash', 'spam')
>>> bayes.train('free money', 'spam')
>>> bayes.train('i will be free on next Monday', 'inbox')
>>> bayes.train('do not forget to send the money', 'inbox')
>>> bayes.train('50$ free poker deposit', 'spam')
>>> 
>>> # Classify terms
>>> bayes.classify('free poker')
'spam'
>>> 
>>> bayes.classify('money')
'spam'
>>> 
>>> bayes.classify('send money')
'inbox'
>>> 
>>> bayes.classify('free deposit')
'spam'
```

## License
MIT License

Copyright (c) 2017 barnabas markus

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
