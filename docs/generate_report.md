# Generate report using `Quarto`

To generate report you need have `Quarto` and `LaTeX` installed on you system, or you can use our dev-container which
has all necessary dependencies encapsulated in docker

Build docker image

```bash
docker build -t gbr-dev -f Dockerfile_dev .
```

Run dev-container

- Use `-it` flag to run container in interactive mode
- Use `--rm` flag to automatically remove container after you close bash session
- Use `-v` flag to mount current directory, so you could edit files in container from you ide
- Use `--net` flag to allow network traffic between host and container (for `preview` mode)

```bash
docker run -it --rm -v $(pwd):/Grocery-Basket-Recommender --net=host gbr-dev
```

Once in container, you can run this command to create `pdf` file of report

```bash
quarto render report.qmd
```

## Live preview

Instead of running `render` command every time you change document, try `preview` mode

```bash
quarto preview report.qmd
```

This command will start web server in container which will re-render document every time you change file.
You will see url in logs
