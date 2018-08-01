def generate_html(image_id, topic, target_captions, generated_captions):
    with open("{}.html".format(image_id), "w") as f:
        f.write("<!DOCTYPE html>")
        f.write("<html>")
        f.write("<body>")
        f.write("<h1>{}</h1>".format(image_id))
        f.write("<img src=\"{}\".jpg height=\"400\" width=\"400\">".format(image_id))
        f.write("<h2>Topic: {}</h2>".format(topic))
        f.write("<h2>Target Captions</h2>")
        f.write("<p>{}</p>".format(target_captions))
        f.write("<h2>Generated Captions</h2>")
        f.write("<ol>")
        for caption in generated_captions:
            f.write("<li>{}</li>".format(caption))
        f.write("</ol>")
        f.write("</body>")
        f.write("</html>")
