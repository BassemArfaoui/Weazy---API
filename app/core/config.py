import os
from pymilvus import  connections , exceptions


DOWNLOADS_DIR = "downloads"

SUCCESS_MESSAGES = [
    "We found some great product matches for you!",
    "Here are a few items we think you'll like based on your image.",
    "These results were picked just for you.",
    "We’ve matched your image with the following products.",
    "Take a look at these options, we think they’re a great fit.",
    "You might like these items we found from your search.",
    "Here’s what we found from your image.",
    "These products look like a strong match for your image.",
    "We’ve handpicked some visually similar items you might love.",
    "These are our top results based on your image.",
    "We think these styles closely match what you’re looking for.",
    "We analyzed your image and found these product matches.",
    "We pulled together the most relevant results for you.",
    "Here’s what we came up with from your upload!",
    "These picks were selected to match your image.",
    "We hope these results help you find what you're looking for.",
    "Here are some visually similar styles you may want to explore.",
    "These options are the closest we could find to your image.",
    "We've put together a few results that match your style."
]



ERROR_MESSAGES = [
    "We couldn’t find any matching products. Want to try a different image?",
    "Sorry, we didn’t find anything similar this time. Try another image?",
    "We looked, but couldn’t find a close match. Feel free to upload a new photo.",
    "It seems we weren’t able to match your image. A clearer photo might help.",
    "We couldn't identify similar items, try a different angle or background.",
    "No matches this time. You can try again or reach out at test@gmail.com for help.",
    "Looks like nothing came up. Please try another image—we’re happy to help.",
    "We didn’t find anything close. Want to give it another shot?",
    "Unfortunately, we didn’t find a match. Try a clearer or closer image?",
    "We weren’t able to find products for that image. Want to try uploading another one?",
    "That one didn’t return results. Maybe try a new image or contact us for assistance.",
    "We weren’t able to match your image just now. Let’s try a different one.",
    "We tried, but didn’t find any results. A new photo might do the trick.",
    "We couldn’t recognize enough to find a match. Please try again with a sharper image.",
    "Your image didn’t return results, don’t hesitate to reach out if you need help.",
    "Nothing came up from that search. Would you like to try again with a different image?",
    "We’re sorry, we couldn’t find similar products. A different image may help.",
    "Sorry, we didn’t find a match this time. Our team is here if you need support.",
    "We scanned the image but didn’t find any matches. Another picture might yield better results.",
    "We looked through our catalog but didn’t find a match. Try another image or get in touch with us."
]




def connectToMilvus(uri, token):
    try:
        connections.connect(
            alias="default",
            uri=uri,
            token=token
        )
        print("Connected to Milvus")
    except exceptions.MilvusException as e:
        print(f"Failed to connect to Milvus: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while connecting to Milvus: {e}")



os.makedirs(DOWNLOADS_DIR, exist_ok=True)

