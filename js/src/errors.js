/** A bad argument to a library function. The CLI reports it as exit 2. */
export class ValueError extends Error {
  constructor(message) {
    super(message);
    this.name = "ValueError";
  }
}

/**
 * The separation check could not run: no speech windows, no gaps between them, or (with an offset)
 * no window inside the mix. The CLI reports it as exit 2, not as a failing mix.
 */
export class NoSpeechError extends ValueError {
  constructor(message) {
    super(message);
    this.name = "NoSpeechError";
  }
}

/** ffmpeg is missing, or the file could not be read. The message is one sentence for the CLI. */
export class DecodeError extends Error {
  constructor(message) {
    super(message);
    this.name = "DecodeError";
  }
}
