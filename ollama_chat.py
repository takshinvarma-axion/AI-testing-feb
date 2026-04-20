# Visit https://ollama.com/ to download Ollama

# visit https://ollama.com/library to find models from library

# ollama 
# ollama pull llama3
# ollama run llama3 --> for CLI Interaction.
# ollama serve --> for API interaction.
# ollama list --> list the models you have downloaded

import ollama

response = ollama.chat(model='llama3',
            messages=[
                {'role':'user','content': 'What is AI? how is it different from ML and DL?'}
            ])

print(response['message']['content'])