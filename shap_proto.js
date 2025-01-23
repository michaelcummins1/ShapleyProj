function calculateShapleyValues(model, inputData, baseline, numSamples = 100) {
    const featureCount = inputData.length;
    let shapleyValues = Array(featureCount).fill(0);

    for (let i = 0; i < featureCount; i++) {
        let marginalContributions = [];
        for (let sample = 0; sample < numSamples; sample++) {
            const permutedIndices = shuffle([...Array(featureCount).keys()]);
            let coalition = [];
            let withoutFeature = baseline.slice();
            let withFeature = baseline.slice();

            for (let j = 0; j < permutedIndices.length; j++) {
                if (permutedIndices[j] === i) {
                    withFeature[i] = inputData[i];
                    break;
                }
                coalition.push(permutedIndices[j]);
                withoutFeature[permutedIndices[j]] = inputData[permutedIndices[j]];
                withFeature[permutedIndices[j]] = inputData[permutedIndices[j]];
            }

            const outputWith = model.predict(withFeature);
            const outputWithout = model.predict(withoutFeature);
            const marginal = outputWith - outputWithout;
            marginalContributions.push(marginal);
        }

        shapleyValues[i] = marginalContributions.reduce((sum, x) => sum + x, 0) / numSamples;
    }

    return shapleyValues;
}

function shuffle(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
}