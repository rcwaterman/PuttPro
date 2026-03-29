/**
 * Off-main-thread JPEG encoder for PuttPro mobile client.
 *
 * Receives an ImageBitmap transferred from the main thread, encodes it to
 * JPEG via OffscreenCanvas, and transfers the resulting ArrayBuffer back.
 * Running this in a Worker keeps the main thread free for video capture and
 * UI updates, removing the encoding step as a frame-rate bottleneck.
 */
self.onmessage = async ({ data: { id, bitmap, quality } }) => {
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    canvas.getContext('2d').drawImage(bitmap, 0, 0);
    bitmap.close(); // release GPU-side memory immediately

    const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality });
    const buffer = await blob.arrayBuffer();

    // Transfer ownership of the ArrayBuffer — zero-copy back to main thread.
    self.postMessage({ id, buffer }, [buffer]);
};
