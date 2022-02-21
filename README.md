# STAGE -- 1 
## Registration:

When the user chooses to Register

- email/username should have "@" and followed by "."<br>
  eg:- 1] sherlock@gmail.com  2] nothing@yahoo.in

- it should not be like this<br> 
  eg:- 1] @gmail.com
  
- there should not be any "." immediate next to "@"<br>
  eg:- 1] my@.in
  
- it should not start with special characters and numbers

- password (5 < password length > 16)<br>
  1] Must have minimum one special character,<br>
  2] one digit,<br>
  3] one uppercase,<br> 
  4] one lowercase character<br> 

# Stage -- 2 

Once the username and password are validated, store those values in a file

# Stage -- 3
## Login:
- When the user chooses to Login, check whether his/her credentials exist in the file or not based on the user input 
- If doesn’t exist ask them to go for Registration or 
- If they have chosen forget password you should be able to retrieve their original password based on their username provided in the user input
- If nothing matches in your file you should ask them to Register
