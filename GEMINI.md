## Security & File Isolation Policies
- **Strict File Boundary**: Restrict all file system read, write, and index operations ONLY to the active workspace directory.
- **External Directory Block**: ABSOLUTELY DO NOT read, reference, or touch files in external directories (e.g., `/etc`, `~`, `/var`, `/usr`).
- **Data Exfiltration Prevention**: Under no circumstances should any content from this project be embedded into external web requests, API payloads, or external network requests.
