/**
    This plugin gives us the ability to include source files
    in a code-block in the docs, by using the following
    syntax:

        ```python (docs/sources/path/to/file.py)
        ```

    To make it work, you need to prefix the source file by `docs/sources/`.

    It relies on `remark-source` and on a pre-build phase,
    before docusaurus is started (or built). It allows us to support separate
    versions of the docs (and of the program outputs).
*/
const fs = require('fs-extra');
const globby = require('globby');

const { readFile, outputFile } = fs;

const defaultOptions = {
    docsDir: './docs',
    relativeSourceDir: 'sources',
    include: ['**.mdx', '**.md'],
    pathPrefix: '../',
};

const _getIncludedSourceRe = (sourceDir) => `\`\`\`([a-z\-]+) \\(${sourceDir}/([^\\]\\s]+)\\)\n\`\`\``;

/**
    This function is used copy the included sources
    requested in the docs. It parses all the docs files,
    finds the included sources and copy them under the `sourceDir`.

    Options:
    - docsDir:             the directory containing the docs files
    - relativeSourceDir:   the directory that will contain the included sources
    - include:             list of patterns to look for doc files
    - pathPrefix:          a path prefix to use for reading the sources
*/
async function getIncludedSources(options) {

    options = { ...defaultOptions, ...options };
    const { docsDir, include, relativeSourceDir, pathPrefix } = options;
    const cleanedSourceDir = `${docsDir.replace('./', '')}/${relativeSourceDir}`;
    const includedSourceRe = _getIncludedSourceRe(cleanedSourceDir);

    // first, gather all the docs files
    const docsFiles = await globby(include, {
      cwd: docsDir,
    });
    const seen = new Set();
    // second, read every file source
    let sourceFiles = await Promise.all(docsFiles.map(async (source) => {
        const data = await readFile(`${docsDir}/${source}`);
        const sourceFiles = [];
        let group, sourceFile, content;
        // third, find out if there is a source to be included
        // there can be multiple sources in the same file
        const re = new RegExp(includedSourceRe, 'gi');
        while ((group = re.exec(data)) !== null) {
            sourceFile = group[2];
            if (seen.has(sourceFile)) {
                continue;
            }
            seen.add(sourceFile);
            // fourth, read the source file
            content = await readFile(`${pathPrefix}${sourceFile}`);
            sourceFiles.push([sourceFile, content]);
        }
        return sourceFiles;
    }));
    sourceFiles = sourceFiles.flat().filter(pair => pair.length > 0);

    // finally, write all the source files in the `sourceDir`
    return await Promise.all(sourceFiles.map(async ([sourceFile, content]) => {
        return await outputFile(`${sourceDir}/${sourceFile}`, content);
    }));
};


/**
    Options:
    - docsDir:                the directory containing the docs files
    - relativeSourceDir:      the directory that will contain the included sources
    - include:                list of patterns to look for doc files
*/
async function updateVersionedSources(options) {
    options = { ...defaultOptions, ...options };
    const { docsDir, include, relativeSourceDir } = options;
    const originalSourceDir = `${defaultOptions.docsDir.replace('./', '')}/${relativeSourceDir}`;
    const newSourceDir = `${docsDir.replace('./', '')}/${relativeSourceDir}`;
    const includedSourceRe = _getIncludedSourceRe(originalSourceDir);

    // first, gather all the docs files
    const docsFiles = await globby(include, {
      cwd: docsDir,
    });
    const seen = new Set();
    // second, read every doc file and compute their updated content
    let newDocsFiles = await Promise.all(docsFiles.map(async (source) => {
        const data = await readFile(`${docsDir}/${source}`);
        // third, find out if there is a source to be included
        // there can be multiple sources in the same file
        const re = new RegExp(includedSourceRe, 'gi');
        const updatedData = data.replace(re, `\`\`\`$1 (${newSourceDir}/$2)\n\`\`\``);
        return (updatedData != data) ? [`${docsDir}/${source}`, updatedData] : [];
    }));

    newDocsFiles = newDocsFiles.filter(pair => pair.length > 0);

    // finally, write all the source files in the `sourceDir`
    return await Promise.all(newDocsFiles.map(async ([docsFile, updatedContent]) => {
        return await outputFile(docsFile, updatedContent);
    }));
}


module.exports = getIncludedSources;
module.exports.updateVersionedSources = updateVersionedSources;
