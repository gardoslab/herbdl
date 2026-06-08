# Claude on SCC

Tips on getting Claude Code running on SCC interactive desktop in a terminal.

Open the terminal.

```bash
mkdir ~/claude-code
cd ~/claude-code
module load nodejs  # load nodejs if not already loaded in your session
npm install @anthropic-ai/claude-code

# Add this to your ~/.bashrc
alias claude='npx --prefix ~/claude-code claude'
```

When you start the terminal from a desktop interactive session on SCC, you may have to change the terminal encoding to UTF8: terminal/Set Encoding/Unicode/UTF-8.

The terminal encoding seems to work fine if you `ssh` from MacOS terminal to the SCC.

Then you can just type claude from any directory.
