# RAG-Meshcore

For lack of time, this is what I could document.

llmRanker:
- First test of RAG functionality
- does an okay job at using the llms intuition in combination with document search to look for answers
- I'm fairly certain that it only looks at one document though, I forget if I changed that

- How would I make it better,
  - Better ranking system:
    - don't use tinyllama
    - better prompting and better context-based source identification
    - Web-search right now is a placeholder
   
OllamaChatMesh:
- really simple, establishes a connection on a com port if windows or a /dev/tty if otherwise (u have to change this urself)
- there is a character limit which isn't addressed, and it also uses trash tinyllama just for testing and to run fast

- Nothing much to change, its a good proof of concept and it works for it's objective

llmRanker:
- combinational project, combining the first two files, so when you send it a message on your phone it responds not in directly tinyllama (or whatever model you chose) it should rank sources first then scan those sources, then reinput that context into a final prompt.
- This one does address the character limit on messsages and autosplices responses which is nicer
- And the the web functionality is a work in progress, but one step away from being fully functional
  - two ways to go about it, if the way that the code is set up doesn't work there is a python duckduckgo extension that should work
 
How I would make it better:
  - given information found on documents, always rank that higher than other sources on the same info
  - again we hate tinyllama for any decision making
  - there is for sure a more efficient way to structure the decision making in the code.

For any information about imports and functions that were made in the code, just look it up or AI it. it's not rocket science, but it does nlook wierd on first glance
