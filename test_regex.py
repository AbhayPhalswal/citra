from jarvis_router import JarvisRouter
r = JarvisRouter()
result = r.route("turn on the ac")
print(result.message)