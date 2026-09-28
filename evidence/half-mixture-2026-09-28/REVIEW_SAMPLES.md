# All long-form review samples

All 12 longer outputs are reproduced exactly. Four task/key instances, each with ordinary generation, current reference watermarking and the rejected half-mixture candidate. The mixture is not available in the SDK. Ratings are assistant review, not human acceptance.

Detection labels use an uncalibrated frozen reference rule; they are not authorship verdicts. Spanish length failures remain visible.

## Key 0 / weather_english / ordinary

170 words. Rubric: pass. Matching-key threshold: miss. Review ID: `5e8e5dc95717`.

Weather refers to the short-term atmospheric conditions in a specific place and time. It includes elements like temperature, precipitation, humidity, and wind. For example, a day might be sunny and warm, or rainy and cold. These conditions can change rapidly, from one hour to the next, and are what people typically talk about when they discuss the "weather" on a given day.

Climate, on the other hand, describes the average weather conditions in a region over a much longer period, typically 30 years or more. It reflects patterns and trends, such as average temperatures, typical rainfall, and seasonal variations. Climate tells us what to expect over years, not days, and helps us understand broader environmental changes.

It is important to distinguish between the two because a single cold day or a hot spell does not indicate a shift in climate. Climate changes occur gradually and are influenced by complex factors like greenhouse gas emissions and ocean currents. Understanding this difference helps avoid misunderstandings when discussing environmental issues or weather events.

Review: Three English paragraphs within 150–200 words; correct short-term weather/long-term climate distinction and limitation of single weather events. No invented named person/place, actual measurement or specific date.

## Key 0 / weather_english / reference

160 words. Rubric: pass. Matching-key threshold: hit. Review ID: `3a754cd37ae8`.

Weather refers to the short-term atmospheric conditions in a specific place at a specific time. It includes elements like temperature, humidity, precipitation, wind, and cloud cover. For example, a sunny afternoon or a sudden thunderstorm are both examples of weather. These conditions can change rapidly, sometimes from hour to hour or day to day.

Climate, on the other hand, describes the average weather conditions in a region over a much longer period, typically 30 years or more. It reflects patterns and trends, such as average rainfall, seasonal temperatures, and the frequency of extreme weather events. Climate gives us a broader picture of what to expect in terms of weather over time.

A single cold day or a hot summer does not change the climate. Climate is about long-term trends, not individual events. Understanding this distinction is important for making sense of environmental changes and weather forecasts. Recognizing the difference helps people avoid misinterpreting short-term variations as long-term shifts in climate.

Review: Three English paragraphs within 150–200 words; correct short-term weather/long-term climate distinction and limitation of single weather events. No invented named person/place, actual measurement or specific date.

## Key 0 / weather_english / half_mixture

170 words. Rubric: pass. Matching-key threshold: miss. Review ID: `99b01ee222fa`.

Weather refers to the short-term atmospheric conditions in a specific place at a specific time. It includes factors like temperature, precipitation, humidity, and wind. For example, a sunny day with a high of 75°F or a sudden thunderstorm are all examples of weather. These conditions can change within hours or days and are what people typically report when they talk about the “forecast.”

Climate, on the other hand, describes the average weather conditions in a region over a much longer period, typically 30 years or more. It reflects the typical patterns and trends of temperature, precipitation, and other atmospheric elements. Climate helps us understand what to expect in terms of seasons, typical rainfall, and overall environmental conditions in a particular area.

It’s important to note that a single cold day or a hot week does not indicate a shift in climate. Climate is about long-term trends, not individual events. Understanding this distinction helps avoid confusion between short-term weather changes and the broader, more gradual shifts in our planet’s climate system.

Review: Three English paragraphs, 170 words, correct weather/climate distinction and single-day limitation. The 75°F example is explicitly hypothetical, not a claimed measured statistic; 30 years is conventional context, not invented observation.

## Key 1 / library_spanish / ordinary

191 words. Rubric: pass. Matching-key threshold: miss. Review ID: `149d36a0b4da`.

Una biblioteca puede organizar préstamos de libros mediante un sistema que registre la información necesaria para cada préstamo. Para ello, se debe anotar el título del libro, el nombre del usuario y la fecha en que se lo prestó. Además, se debe indicar la fecha límite para la devolución, para que el usuario sepa cuándo debe devolverlo. Este registro ayuda a mantener el control de los libros y facilita la gestión de los préstamos.

Otra opción es permitir la renovación de los libros, siempre que se respeten las reglas de la biblioteca. Esto significa que el usuario puede extender el plazo de préstamo si el libro no está reservado y se cumple con los requisitos establecidos. Sin embargo, la disponibilidad de esta opción depende de las políticas internas de la biblioteca.

Es importante proteger los datos personales del usuario, como su nombre y el título del libro prestado, para garantizar la privacidad. La biblioteca debe asegurarse de que esta información se almacene de manera segura y solo se acceda cuando sea necesario. Al hacerlo, se respetan las normas de seguridad y se fomenta la confianza entre los usuarios y la institución.

Review: Three Spanish paragraphs within 150–200 words; proposed library process records book, borrower and due date, keeps renewal conditional on rules and protects personal data. No invented specific names, dates, prices or fines.

## Key 1 / library_spanish / reference

214 words. Rubric: fail. Matching-key threshold: hit. Review ID: `54881550c0a1`.

Una biblioteca puede organizar préstamos de libros mediante un sistema de registro que incluye el título del libro, el nombre de la persona que lo recibe y la fecha de devolución estimada. Este proceso permite que los usuarios conozcan qué libros están disponibles y cuándo pueden recogerlos. La biblioteca puede ofrecer opciones como préstamos por semana o por mes, dependiendo de sus reglas. Además, se puede permitir la posibilidad de renovar los libros siempre que el usuario cumpla con las condiciones establecidas.

Para garantizar la organización, la biblioteca debe mantener un registro actualizado de los préstamos, que incluya el nombre del usuario, el libro prestado y la fecha en que debe devolverse. Esta información ayuda a gestionar la disponibilidad de los materiales y a notificar a los usuarios si no devuelven los libros a tiempo. También es importante que la biblioteca establezca límites claros sobre las renovaciones, que pueden variar según su política interna.

Es fundamental proteger los datos personales de los usuarios, especialmente al manejar información sensible como nombres y fechas de préstamo. La biblioteca debe asegurarse de que los registros se almacenen de manera segura y solo sean accesibles por personal autorizado. Al hacerlo, se respeta la privacidad de los usuarios y se fomenta un entorno confiable para el préstamo de libros.

Review: Spanish content covers the requested library process, renewal rules and privacy, but 214 words exceeds the fixed 200-word limit. All three paragraphs retained; no truncation or automatic repair.

## Key 1 / library_spanish / half_mixture

208 words. Rubric: fail. Matching-key threshold: miss. Review ID: `6f1461074905`.

Una biblioteca puede organizar préstamos de libros mediante un sistema que permita registrar los detalles de cada préstamo. Para ello, se puede establecer que cada libro tenga un código único, que se anote junto con el nombre del usuario y la fecha de préstamo. Esto ayuda a llevar un control claro de qué libro se presta y a quién. Además, se puede fijar una fecha límite de devolución, que el usuario debe respetar para garantizar el acceso a otros lectores.

En cuanto a la renovación de los préstamos, la biblioteca puede permitir que los usuarios soliciten una extensión siempre que no haya reservas o conflictos. Esta opción depende de las reglas internas de la biblioteca, por lo que se debe informar con anticipación sobre los términos y condiciones. En este caso, la decisión de renovar o no corresponde a la institución, y el usuario debe seguir las indicaciones establecidas.

Es importante también proteger los datos personales de los usuarios. La biblioteca debe garantizar que la información registrada, como nombres o números de identificación, se almacene de manera segura y solo se acceda cuando sea necesario. Esto contribuye a mantener la privacidad de los lectores y a cumplir con estándares de seguridad y ética en el manejo de información.

Review: Spanish content covers the requested library process, renewal rules and privacy, but 208 words exceeds the fixed 200-word limit. All three paragraphs retained; no truncation or automatic repair.

## Key 2 / weather_english / ordinary

160 words. Rubric: pass. Matching-key threshold: miss. Review ID: `fd36bbf6f3c0`.

Weather refers to the short-term atmospheric conditions in a specific place and time. These include elements like temperature, precipitation, humidity, and wind. Weather can change rapidly, from sunny skies to a sudden storm within hours. It is what you experience on a daily basis and is often discussed in news reports or forecasts.

Climate, on the other hand, describes the average weather conditions in a region over a much longer period, typically 30 years or more. It reflects patterns and trends, such as average temperatures and typical rainfall amounts. Climate is more about what is usually expected in a particular area, rather than what happens on any given day.

A single cold day or a hot week does not indicate a shift in climate. Climate change refers to long-term shifts in these patterns, which can be influenced by various factors, including human activities. Understanding the difference between weather and climate is essential for interpreting environmental changes and making informed decisions.

Review: Three English paragraphs within 150–200 words; correct short-term weather/long-term climate distinction and limitation of single weather events. No invented named person/place, actual measurement or specific date.

## Key 2 / weather_english / reference

173 words. Rubric: pass. Matching-key threshold: hit. Review ID: `18275329ffc2`.

Weather refers to the short-term atmospheric conditions in a specific place and time. It includes elements such as temperature, humidity, precipitation, wind, and cloud cover. Weather can change rapidly, from sunny skies to a thunderstorm in a matter of hours. These changes are influenced by local conditions and can be predicted for a few days ahead with varying degrees of accuracy.

Climate, on the other hand, describes the long-term patterns of weather in a region. It is based on average conditions observed over decades or even centuries. Climate includes not only temperature but also factors like rainfall, wind patterns, and seasonal variations. Understanding climate helps us predict general trends and prepare for long-term environmental changes.

It is important to distinguish between the two, as a single cold day or a warm week does not indicate a shift in climate. Climate change refers to gradual shifts in long-term patterns, which can be influenced by natural processes and human activities. Recognizing this difference helps avoid misinterpretations of short-term weather events as evidence of climate trends.

Review: Three English paragraphs within 150–200 words; correct short-term weather/long-term climate distinction and limitation of single weather events. No invented named person/place, actual measurement or specific date.

## Key 2 / weather_english / half_mixture

168 words. Rubric: pass. Matching-key threshold: miss. Review ID: `e2b8097f7654`.

Weather refers to the short-term atmospheric conditions in a specific place and time. It includes elements such as temperature, humidity, precipitation, wind, and cloud cover. Weather can change rapidly, from sunny skies to a thunderstorm in just a few hours. These variations are influenced by local conditions and can be unpredictable from day to day.

Climate, on the other hand, describes the average weather conditions in a region over a much longer period, typically 30 years or more. It reflects the typical patterns of temperature, precipitation, and other atmospheric factors that define a region's overall character. Climate helps us understand what to expect in terms of seasons, rainfall, and extreme weather events over time.

A single cold day or a warm spell does not indicate a shift in climate. Climate changes occur gradually and are influenced by complex factors such as ocean currents, atmospheric composition, and natural cycles. Understanding the difference between weather and climate is essential for interpreting environmental changes and making informed decisions about the future.

Review: Three English paragraphs within 150–200 words; correct short-term weather/long-term climate distinction and limitation of single weather events. No invented named person/place, actual measurement or specific date.

## Key 3 / library_spanish / ordinary

207 words. Rubric: fail. Matching-key threshold: miss. Review ID: `c0109ad25d14`.

Una biblioteca puede organizar préstamos de libros utilizando un sistema informático que permite registrar los detalles de cada préstamo. En este sistema, se anota el título del libro, su autor y el código de identificación, junto con el nombre y la dirección del usuario. Esto ayuda a gestionar los préstamos de manera eficiente y a mantener un registro de los materiales disponibles. Además, se marca la fecha en que el libro se presta y la fecha límite de devolución, para que los usuarios sepan cuándo debe devolverlo.

Para garantizar un buen uso de los recursos, la biblioteca puede establecer reglas sobre la renovación de los préstamos. Esto significa que los usuarios pueden solicitar la extensión del préstamo siempre que haya disponibilidad, siempre que las reglas de la biblioteca lo permitan. Esta opción ayuda a evitar que los libros estén bloqueados por largos períodos y a mantener un acceso justo para todos los lectores.

Es importante que la biblioteca proteja los datos personales de los usuarios. Por lo tanto, solo se registran los datos necesarios para gestionar el préstamo, como nombre y dirección, y se garantiza su confidencialidad. Esta medida asegura que los usuarios se sientan seguros al utilizar los servicios de la biblioteca y respeta su privacidad.

Review: Spanish content covers the requested library process, renewal rules and privacy, but 207 words exceeds the fixed 200-word limit. All three paragraphs retained; no truncation or automatic repair.

## Key 3 / library_spanish / reference

172 words. Rubric: pass. Matching-key threshold: miss. Review ID: `7b0a9d92e4be`.

Una biblioteca puede organizar préstamos de libros mediante un sistema que permite registrar los detalles de cada préstamo. Para ello, se puede establecer un registro que incluya el título del libro, el nombre de la persona que lo recibe y la fecha en que se debe devolver. Así, la biblioteca mantiene un control sobre los libros y asegura que estén disponibles para otros usuarios.

Además, se puede permitir la renovación de los préstamos, siempre que el sistema lo permita. Esto significa que el usuario puede solicitar una extensión del préstamo siempre que los registros indiquen que el libro no ha sido reservado por otra persona. Las reglas sobre renovaciones pueden variar según las normas de la biblioteca.

Es importante también proteger los datos personales de los usuarios. La biblioteca debe garantizar que la información como nombres o direcciones no se comparta con terceros sin autorización. Esto asegura la privacidad de quienes utilizan los servicios. En resumen, el sistema de préstamos debe ser claro, accesible y respetuoso con la confidencialidad de los usuarios.

Review: Three Spanish paragraphs within 150–200 words; proposed library process records book, borrower and due date, keeps renewal conditional on rules and protects personal data. No invented specific names, dates, prices or fines.

## Key 3 / library_spanish / half_mixture

212 words. Rubric: fail. Matching-key threshold: miss. Review ID: `c79b74602b1f`.

Una biblioteca puede organizar préstamos de libros mediante un sistema de registro que permite seguir el historial de cada préstamo. En este sistema, se anota qué libro se presta, quién lo recibe y la fecha en que debe devolverse. Esto ayuda a mantener un control ordenado y facilita la gestión de las solicitudes. Se puede optar por un sistema físico, como tarjetas o listas, o digital, utilizando software especializado que simplifica la organización.

Además, las bibliotecas pueden ofrecer la opción de renovar los libros, siempre que se respeten las reglas establecidas. Esta renovación depende del horario de la biblioteca y de la disponibilidad del libro. Es importante que los usuarios conozcan estas condiciones para evitar inconvenientes al momento de la devolución. También se debe considerar establecer límites claros para evitar la acumulación de materiales prestados por parte de los usuarios.

Por último, es fundamental proteger los datos personales de los usuarios. La biblioteca debe garantizar que la información de los préstamos se maneje con confidencialidad, respetando la privacidad de los lectores. Esto puede lograrse mediante medidas como el uso de identificadores anónimos o la limitación del acceso a los registros solo para personal autorizado. Estas prácticas contribuyen a mantener la seguridad y la confianza de los usuarios en el servicio de préstamo.

Review: Spanish content covers the requested library process, renewal rules and privacy, but 212 words exceeds the fixed 200-word limit. All three paragraphs retained; no truncation or automatic repair.
