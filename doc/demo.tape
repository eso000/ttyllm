# 1. Output Configurations
Output assets/readme_demo.gif

# 2. Require your app to be installed to run the tape
Require ttyllm

# 3. Environment & UI Settings
Set Shell "bash"
Set FontSize 18
Set Width 900
Set Height 500
Set Padding 20
Set Theme "Dracula"
Set WindowBar Colorful
Set TypingSpeed 45ms

# 4. Preparation (Hide setup from final GIF)
Hide
Type "export PS1='$ '" Enter
Type "clear" Enter
Show

# 5. Launch the TUI app
Type "ttyllm" Enter
Enter
Sleep 1.5s

# 6. Execute the Demo routine
Type "Hello! Explain what a Python standard library is in one short sentence." Enter
Sleep 7s

# Open Help and scroll down
Type ":h"
Sleep 1s
Enter
Sleep 2
Down 9
Sleep 2s

Type ":n"
Sleep 1s
Enter
Type ":s"
Sleep 1s
Enter
Type ":r"
Enter
Sleep 7s

Type ":bp"
Sleep 1s
Enter
Sleep 2s

# Quit the app safely
Type ":q"
Enter
Sleep 1s

