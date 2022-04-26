# Search Engine
This is project created as part of technical interview for Smart Cat. Semantic similarity search based on TF-IDF and bag of words algorithms.

## Requirements

For building and running the application you need python libraries (you can install them with pip):

* python >= 3.8

- numpy==1.20.0
- tqdm==4.61.0


#### OR
```shell
pip install -r requirements.txt
```

## Running the application locally

```shell
python run.py
```

## Getting Started

### Installing

Clone the repository. You must have some python package management environment.

```bash
git clone https://github.com/nemanja1995/SmartCatSearchEngine.git
cd SmartCatSearchEngine
pip install -r requirements.txt
```

### Examples
When you first run `run.py` it will process whole corpus and save cached data to `\cached` 
directory. Every next time it will use cached data instead of processing corpus again.

#### Execution example:
```text
$ python run.py

> enter query: 
what is array type in python
--------------------
(0.8716, 'How do I extract a type from an array in typescript?')
(0.8716, 'Typescript: How to map over union array type?')
(0.8716, 'Adding 2 array `type`s in Golang')
(0.8716, 'Assignment to expression with array type AND request for member in something not a structure or union')
(1.0, 'Python: setting type of numpy structure array')



> enter query: 
quit
```

#### More arguments
You can give arguments to script. Options:
* (-v) --vector_size - Size of embedding vectors
* (-c) --corpus_path - Path to corpus
* (-s) --stop_words_path - Path to file with stopwords
* (-f) --force_process - Force engine to process corpus again
* (-dp) --qse_data_path - Path to cached data for question search engine

You can use `-h` or `--help` form more info about arguments

## Api (bonus)

### API Endpoint : http://localhost:8081

Start service
```bash
python web_server.py
```

##### Checks if service is up
* Method : `GET`
* Content-Type: `application/json`
  
* Send request example:
```curl
curl --location --request GET 'http://localhost:8081'
```

* Response JSON example
```json
{
    "message": "ping!",
    "status": "OK"
}
```

##### Send query request
* Method : `POST`
* Content-Type: `application/json`
  
* Send request example:
```curl
curl --location --request POST 'http://localhost:8081' \
--header 'Content-Type: text/plain' \
--data-raw '{
    "questions": ["how to make sure a file'\''s integrity in C#?"]
}'
```

* Response JSON example
```json
[
	{
		"question": "how to make sure a file's integrity in C#?",
		"similar_questions": [
			[
				1.0,
				"\"no newline at end of file\" C"
			],
			[
				1.0,
				"How can I use Rails 5.2 credentials in capistrano's deploy.rb file?"
			],
			[
				1.0,
				"VSCode: Prevent split editor to open same file left & right"
			],
			[
				1.0,
				"How to move file or directories in linux"
			],
			[
				1.0,
				"Reading and displaying a file in Cobol"
			]
		]
	}
]
```

## Authors
* **Nemanja Janjic** - nemanja1995ng@gmail.com
