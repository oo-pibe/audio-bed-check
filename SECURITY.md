# Security

Report a vulnerability privately through GitHub's "Report a vulnerability" button on this
repository, or by email to contact@roadtokickoff.com. Please do not open a public issue for it.

The package itself makes no network requests. It runs ffmpeg on the files you name, with the input
restricted to the local file and ffmpeg's protocol whitelist set to file and pipe, so a filename that
looks like a URL is still read as a file. ffmpeg parses the file, so ffmpeg's demuxer bugs apply; keep
it updated. One environment variable is read, AUDIO_BED_CHECK_FFMPEG.
